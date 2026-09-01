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
from utils.tools import save_args_to_json


class Exp_Classification(Exp_Basic):
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _criterion(self, train_data):
        counts = np.asarray(train_data.class_counts, dtype=np.float64)
        if np.any(counts == 0):
            raise ValueError(f"Every class needs training windows, got {counts.tolist()}")
        weights = counts.sum() / (len(counts) * counts)
        return nn.CrossEntropyLoss(
            weight=torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        )

    @torch.inference_mode()
    def _evaluate(self, loader, criterion):
        self.model.eval()
        losses = []
        actual = []
        predicted = []
        probabilities = []
        for windows, labels in loader:
            windows = windows.float().to(self.device, non_blocking=True)
            labels = labels.long().to(self.device, non_blocking=True)
            logits = self.model(windows, None, None, None)
            losses.append(float(criterion(logits, labels)) * len(labels))
            probs = torch.softmax(logits, dim=-1)
            actual.append(labels.cpu().numpy())
            predicted.append(probs.argmax(dim=-1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())
        actual = np.concatenate(actual)
        predicted = np.concatenate(predicted)
        probabilities = np.concatenate(probabilities)
        return {
            "loss": float(sum(losses) / len(actual)),
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro")),
            "weighted_f1": float(f1_score(actual, predicted, average="weighted")),
            "actual": actual,
            "predicted": predicted,
            "probabilities": probabilities,
        }

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
            optimizer, mode="max", factor=0.5, patience=1
        )
        use_amp = bool(self.args.use_amp and self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        checkpoint_dir = Path(self.args.checkpoints) / setting
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint.pth"

        best_macro_f1 = -np.inf
        best_state = None
        stale_epochs = 0
        history = []
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
                    logits = self.model(windows, None, None, None)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(labels)
                example_count += len(labels)

            validation = self._evaluate(validation_loader, criterion)
            train_loss = loss_sum / max(example_count, 1)
            scheduler.step(validation["macro_f1"])
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
            }
            history.append(row)
            print(
                f"Epoch {epoch:02d}/{self.args.train_epochs} "
                f"train_loss={train_loss:.5f} "
                f"val_accuracy={validation['accuracy']:.4f} "
                f"val_macro_F1={validation['macro_f1']:.4f} "
                f"time={time.time() - epoch_start:.1f}s",
                flush=True,
            )
            if validation["macro_f1"] > best_macro_f1 + 1e-6:
                best_macro_f1 = validation["macro_f1"]
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, checkpoint_path)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.args.patience:
                    print("Early stopping", flush=True)
                    break

        if best_state is None:
            raise RuntimeError("Classification training produced no checkpoint")
        self.model.load_state_dict(best_state)
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
        result = self._evaluate(test_loader, criterion)
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
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream, ensure_ascii=False, indent=2)
        np.savez_compressed(
            output_dir / "predictions.npz",
            actual=result["actual"],
            predicted=result["predicted"],
            probabilities=result["probabilities"],
        )
        save_args_to_json(self.args, output_dir / "args.json")

        print(
            f"TEST accuracy={result['accuracy']:.4f} "
            f"macro_F1={result['macro_f1']:.4f} "
            f"weighted_F1={result['weighted_f1']:.4f}",
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
