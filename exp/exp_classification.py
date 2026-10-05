"""Training and evaluation loop for four-class indoor-fire source classification."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.evidential_loss import evidential_classification_loss
from utils.tools import save_args_to_json


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017).

    Unlike static per-class weighting, the (1 - p_t)^gamma factor scales
    each SAMPLE's loss by how wrong the model currently is on it: easy,
    already-confident predictions contribute almost nothing to the
    gradient, while hard/ambiguous samples (e.g. Nuisance-Fire boundary
    cases) keep driving learning throughout training.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = targets.long()
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * log_p_t
        if self.weight is not None:
            loss = loss * self.weight.gather(0, targets)
        return loss.mean()


class Exp_Classification(Exp_Basic):
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _criterion(self, train_data):
        override = getattr(self.args, "class_weight_override", None)
        if override is not None:
            weights = np.asarray(override, dtype=np.float64)
            if len(weights) != self.args.num_class:
                raise ValueError(
                    f"--class_weight_override must have {self.args.num_class} values, "
                    f"got {len(weights)}"
                )
        else:
            counts = np.asarray(train_data.class_counts, dtype=np.float64)
            if np.any(counts == 0):
                raise ValueError(f"Every class needs training windows, got {counts.tolist()}")
            class_weight_power = float(
                getattr(self.args, "class_weight_power", 1.0)
            )
            if not 0.0 <= class_weight_power <= 1.0:
                raise ValueError("--class_weight_power must be between 0 and 1")
            weights = (
                counts.sum() / (len(counts) * counts)
            ) ** class_weight_power
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        if getattr(self.args, "use_focal_loss", False):
            return FocalLoss(weight=weight_tensor, gamma=getattr(self.args, "focal_gamma", 2.0))
        return nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=getattr(self.args, "label_smoothing", 0.0),
        )

    def _loss_and_outputs(self, output, labels, criterion, annealing=1.0):
        if isinstance(output, tuple):
            logits, aux_loss = output
            loss = criterion(logits, labels) + aux_loss
            return loss, torch.softmax(logits, dim=-1), None
        if isinstance(output, dict) and "alpha" in output:
            evidential_loss, _ = evidential_classification_loss(
                output["alpha"],
                labels,
                class_weights=criterion.weight,
                annealing=annealing,
                kl_weight=self.args.kl_weight,
            )
            if "logits" in output:
                loss = criterion(output["logits"], labels) + float(
                    getattr(self.args, "edl_loss_weight", 0.1)
                ) * evidential_loss
            else:
                loss = evidential_loss
            if "aux_loss" in output:
                loss = loss + output["aux_loss"]
            branch_alphas = [
                output[key]
                for key in ("sensor_branch_alpha", "time_branch_alpha")
                if key in output
            ]
            if branch_alphas:
                branch_losses = [
                    evidential_classification_loss(
                        alpha,
                        labels,
                        class_weights=criterion.weight,
                        annealing=annealing,
                        kl_weight=self.args.kl_weight,
                    )[0]
                    for alpha in branch_alphas
                ]
                loss = loss + self.args.branch_loss_weight * torch.stack(
                    branch_losses
                ).mean()
            return loss, output["probabilities"], output["uncertainty"]
        loss = criterion(output, labels)
        return loss, torch.softmax(output, dim=-1), None

    @torch.inference_mode()
    def _evaluate(self, loader, criterion):
        self.model.eval()
        losses = []
        actual = []
        predicted = []
        probabilities = []
        uncertainties = []
        for windows, labels in loader:
            windows = windows.float().to(self.device, non_blocking=True)
            labels = labels.long().to(self.device, non_blocking=True)
            output = self.model(windows, None, None, None)
            loss, probs, uncertainty = self._loss_and_outputs(
                output, labels, criterion
            )
            losses.append(float(loss) * len(labels))
            actual.append(labels.cpu().numpy())
            predicted.append(probs.argmax(dim=-1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())
            if uncertainty is not None:
                uncertainties.append(uncertainty.squeeze(-1).cpu().numpy())
        actual = np.concatenate(actual)
        predicted = np.concatenate(predicted)
        probabilities = np.concatenate(probabilities)
        result = {
            "loss": float(sum(losses) / len(actual)),
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro")),
            "weighted_f1": float(f1_score(actual, predicted, average="weighted")),
            "actual": actual,
            "predicted": predicted,
            "probabilities": probabilities,
        }
        result["uncertainties"] = (
            np.concatenate(uncertainties) if uncertainties else None
        )
        return result

    def train(self, setting):
        train_data, train_loader = self._get_data("train")
        validation_data, validation_loader = self._get_data("val")
        criterion = self._criterion(train_data)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.args.lr_factor,
            patience=self.args.lr_patience,
            min_lr=self.args.min_learning_rate,
        )
        use_amp = bool(self.args.use_amp and self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        checkpoint_dir = Path(self.args.checkpoints) / setting
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint.pth"

        best_macro_f1 = -np.inf
        best_epoch = 0
        best_state = None
        stale_epochs = 0
        history = []
        total_train_time = 0.0
        total_train_samples = 0
        for epoch in range(1, self.args.train_epochs + 1):
            self.model.train()
            epoch_start = time.time()
            loss_sum = 0.0
            example_count = 0
            for windows, labels in train_loader:
                windows = windows.float().to(self.device, non_blocking=True)
                labels = labels.long().to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=use_amp,
                ):
                    output = self.model(windows, None, None, None)
                    annealing = min(
                        1.0,
                        epoch / max(1, self.args.annealing_epochs),
                    )
                    loss, _, _ = self._loss_and_outputs(
                        output,
                        labels,
                        criterion,
                        annealing=annealing,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(labels)
                example_count += len(labels)

            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time
            total_train_samples += example_count
            train_throughput = example_count / max(epoch_time, 1e-6)

            validation = self._evaluate(validation_loader, criterion)
            train_loss = loss_sum / max(example_count, 1)
            scheduler.step(validation["macro_f1"])
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "epoch_time_s": epoch_time,
                "train_throughput_samples_per_s": train_throughput,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"Epoch {epoch:02d}/{self.args.train_epochs} "
                f"train_loss={train_loss:.5f} "
                f"val_accuracy={validation['accuracy']:.4f} "
                f"val_macro_F1={validation['macro_f1']:.4f} "
                f"time={epoch_time:.1f}s "
                f"throughput={train_throughput:.1f} samples/s",
                flush=True,
            )
            # Always retain the globally best validation model.  min_epochs
            # controls only when early-stopping patience starts counting.
            if validation["macro_f1"] > best_macro_f1 + 1e-6:
                best_macro_f1 = validation["macro_f1"]
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, checkpoint_path)
                stale_epochs = 0
            elif epoch >= self.args.min_epochs:
                stale_epochs += 1
                if stale_epochs >= self.args.patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(min_epochs={self.args.min_epochs}, "
                        f"patience={self.args.patience})",
                        flush=True,
                    )
                    break

        if best_state is None:
            raise RuntimeError("Classification training produced no checkpoint")
        self.model.load_state_dict(best_state)
        avg_throughput = total_train_samples / max(total_train_time, 1e-6)
        print(
            f"\nTraining summary: total_time={total_train_time:.1f}s "
            f"avg_throughput={avg_throughput:.1f} samples/s "
            f"device={self.device}\n"
            f"Best validation model: epoch={best_epoch} "
            f"macro_F1={best_macro_f1:.4f}",
            flush=True,
        )
        history_path = checkpoint_dir / "history.json"
        with history_path.open("w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)
        self.class_names = tuple(train_data.class_names)
        self.split_summary = getattr(
            train_data,
            "event_summary",
            getattr(train_data, "split_days", None),
        )
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data("test")
        if test:
            checkpoint = getattr(self.args, "checkpoint", None)
            checkpoint_path = (
                Path(checkpoint)
                if checkpoint
                else Path(self.args.checkpoints) / setting / "checkpoint.pth"
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )

        criterion = nn.CrossEntropyLoss()
        infer_start = time.time()
        result = self._evaluate(test_loader, criterion)
        infer_time = time.time() - infer_start
        total_test_samples = len(test_data)
        infer_throughput = total_test_samples / max(infer_time, 1e-6)
        ms_per_sample = 1000.0 * infer_time / max(total_test_samples, 1)
        print(
            f"Inference: {total_test_samples} samples in {infer_time:.3f}s "
            f"({infer_throughput:.1f} samples/s, {ms_per_sample:.4f} ms/sample) "
            f"device={self.device}",
            flush=True,
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            result["actual"],
            result["predicted"],
            labels=np.arange(self.args.num_class),
            zero_division=0,
        )
        class_names = tuple(test_data.class_names)
        per_class = {
            class_names[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(class_names))
        }
        matrix = confusion_matrix(
            result["actual"], result["predicted"], labels=np.arange(self.args.num_class)
        )
        output_dir = Path(self.args.results) / setting
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
            "class_names": class_names,
            "per_class": per_class,
            "confusion_matrix": matrix.tolist(),
            "data_split": getattr(
                test_data,
                "event_summary",
                getattr(test_data, "split_days", None),
            ),
            "inference_time_s": infer_time,
            "inference_throughput_samples_per_s": infer_throughput,
            "inference_ms_per_sample": ms_per_sample,
            "device": str(self.device),
        }
        if result["uncertainties"] is not None:
            uncertainty = result["uncertainties"]
            correct = result["actual"] == result["predicted"]
            metrics.update(
                {
                    "mean_uncertainty": float(uncertainty.mean()),
                    "correct_uncertainty": float(uncertainty[correct].mean()),
                    "incorrect_uncertainty": float(
                        uncertainty[~correct].mean()
                    ) if (~correct).any() else 0.0,
                }
            )
            for index, name in enumerate(class_names):
                class_mask = result["actual"] == index
                per_class[name]["mean_uncertainty"] = float(
                    uncertainty[class_mask].mean()
                )
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream, ensure_ascii=False, indent=2)
        prediction_data = {
            "actual": result["actual"],
            "predicted": result["predicted"],
            "probabilities": result["probabilities"],
        }
        if result["uncertainties"] is not None:
            prediction_data["uncertainties"] = result["uncertainties"]
        np.savez_compressed(output_dir / "predictions.npz", **prediction_data)
        save_args_to_json(self.args, output_dir / "args.json")

        print(
            f"TEST accuracy={result['accuracy']:.4f} "
            f"macro_F1={result['macro_f1']:.4f} "
            f"weighted_F1={result['weighted_f1']:.4f}",
            flush=True,
        )
        if result["uncertainties"] is not None:
            print(
                f"  uncertainty mean={metrics['mean_uncertainty']:.4f} "
                f"correct={metrics['correct_uncertainty']:.4f} "
                f"incorrect={metrics['incorrect_uncertainty']:.4f}",
                flush=True,
            )
        for name in class_names:
            values = per_class[name]
            print(
                f"  {name:<8} F1={values['f1']:.4f} "
                f"recall={values['recall']:.4f} "
                f"precision={values['precision']:.4f} "
                f"support={values['support']}",
                flush=True,
            )
        print("Confusion matrix:")
        print(matrix)
        print(f"Saved results: {output_dir}")
        return metrics
