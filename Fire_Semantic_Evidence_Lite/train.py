"""Train the lightweight level-trend semantic evidence classifier."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.data_loader import Dataset_IndoorFireTernary  # noqa: E402
from Fire_Semantic_Evidence_Lite.losses import (  # noqa: E402
    evidential_classification_loss,
)
from Fire_Semantic_Evidence_Lite.model import (  # noqa: E402
    FireSemanticEvidenceLite,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight semantic evidence network for fire states"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT
        / "dataset"
        / "Indoor Fire Dataset with Distributed Multi-Sensor Nodes.csv",
    )
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-gap-seconds", type=float, default=20.0)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--evidence-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--branch-loss-weight", type=float, default=0.2)
    parser.add_argument("--annealing-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results_fire_dual_semantic_evidence_lite",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_datasets(args):
    dataset_args = SimpleNamespace(
        data_path=args.data.name,
        seq_len=args.seq_len,
        classification_stride=args.stride,
        max_gap_seconds=args.max_gap_seconds,
    )
    return {
        split: Dataset_IndoorFireTernary(
            dataset_args,
            root_path=args.data.parent,
            flag=split,
            size=[args.seq_len, 0, 0],
        )
        for split in ("train", "val", "test")
    }


def build_loader(dataset, args, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=max(0, args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


def calibration_error(probabilities, actual, bins=10):
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    error = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            bin_accuracy = np.mean(predicted[mask] == actual[mask])
            error += mask.mean() * abs(bin_accuracy - confidence[mask].mean())
    return float(error)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    actual = []
    predicted = []
    probabilities = []
    uncertainties = []
    level_uncertainties = []
    trend_uncertainties = []
    level_weights = []
    trend_weights = []
    for windows, labels in loader:
        output = model(windows.float().to(device, non_blocking=True))
        probs = output["probabilities"]
        actual.append(labels.numpy())
        predicted.append(probs.argmax(dim=-1).cpu().numpy())
        probabilities.append(probs.cpu().numpy())
        uncertainties.append(output["uncertainty"].squeeze(-1).cpu().numpy())
        level_uncertainties.append(
            output["level_uncertainty"].squeeze(-1).cpu().numpy()
        )
        trend_uncertainties.append(
            output["trend_uncertainty"].squeeze(-1).cpu().numpy()
        )
        level_weights.append(output["level_weight"].squeeze(-1).cpu().numpy())
        trend_weights.append(output["trend_weight"].squeeze(-1).cpu().numpy())
    actual = np.concatenate(actual)
    predicted = np.concatenate(predicted)
    probabilities = np.concatenate(probabilities)
    uncertainties = np.concatenate(uncertainties)
    level_uncertainties = np.concatenate(level_uncertainties)
    trend_uncertainties = np.concatenate(trend_uncertainties)
    level_weights = np.concatenate(level_weights)
    trend_weights = np.concatenate(trend_weights)
    one_hot = np.eye(probabilities.shape[1])[actual]
    correct = predicted == actual
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted")),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": calibration_error(probabilities, actual),
        "mean_uncertainty": float(uncertainties.mean()),
        "correct_uncertainty": float(uncertainties[correct].mean()),
        "incorrect_uncertainty": float(uncertainties[~correct].mean())
        if (~correct).any()
        else 0.0,
        "mean_level_uncertainty": float(level_uncertainties.mean()),
        "mean_trend_uncertainty": float(trend_uncertainties.mean()),
        "mean_level_weight": float(level_weights.mean()),
        "mean_trend_weight": float(trend_weights.mean()),
        "actual": actual,
        "predicted": predicted,
        "probabilities": probabilities,
        "uncertainties": uncertainties,
        "level_uncertainties": level_uncertainties,
        "trend_uncertainties": trend_uncertainties,
        "level_weights": level_weights,
        "trend_weights": trend_weights,
    }


def make_metrics(evaluation, class_names):
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        evaluation["actual"], evaluation["predicted"], labels=labels, zero_division=0
    )
    metrics = {
        key: evaluation[key]
        for key in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "brier_score",
            "ece",
            "mean_uncertainty",
            "correct_uncertainty",
            "incorrect_uncertainty",
            "mean_level_uncertainty",
            "mean_trend_uncertainty",
            "mean_level_weight",
            "mean_trend_weight",
        )
    }
    metrics["class_names"] = list(class_names)
    metrics["per_class"] = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "mean_uncertainty": float(
                evaluation["uncertainties"][evaluation["actual"] == index].mean()
            ),
            "level_weight": float(
                evaluation["level_weights"][evaluation["actual"] == index].mean()
            ),
            "trend_weight": float(
                evaluation["trend_weights"][evaluation["actual"] == index].mean()
            ),
        }
        for index, name in enumerate(class_names)
    }
    metrics["confusion_matrix"] = confusion_matrix(
        evaluation["actual"], evaluation["predicted"], labels=labels
    ).tolist()
    return metrics


def train_one_run(args, datasets, device, run_index):
    seed = args.seed + run_index
    set_seed(seed)
    train_loader = build_loader(datasets["train"], args, True, seed)
    validation_loader = build_loader(datasets["val"], args, False, seed)
    test_loader = build_loader(datasets["test"], args, False, seed)
    model = FireSemanticEvidenceLite(
        seq_len=args.seq_len,
        channels=len(datasets["train"].feature_names),
        embedding_dim=args.embedding_dim,
        evidence_hidden_dim=args.evidence_hidden_dim,
        num_classes=len(datasets["train"].class_names),
        dropout=args.dropout,
    ).to(device)
    counts = datasets["train"].class_counts.astype(np.float64)
    class_weights = counts.sum() / (len(counts) * counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    best_f1 = -np.inf
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        loss_sum = 0.0
        classification_sum = 0.0
        kl_sum = 0.0
        auxiliary_sum = 0.0
        example_count = 0
        annealing = min(1.0, epoch / max(1, args.annealing_epochs))
        for windows, labels in train_loader:
            windows = windows.float().to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(windows)
            loss, terms = evidential_classification_loss(
                output["alpha"],
                labels,
                class_weights=class_weights,
                annealing=annealing,
                kl_weight=args.kl_weight,
            )
            level_loss, _ = evidential_classification_loss(
                output["level_alpha"],
                labels,
                class_weights=class_weights,
                annealing=annealing,
                kl_weight=args.kl_weight,
            )
            trend_loss, _ = evidential_classification_loss(
                output["trend_alpha"],
                labels,
                class_weights=class_weights,
                annealing=annealing,
                kl_weight=args.kl_weight,
            )
            auxiliary_loss = 0.5 * (level_loss + trend_loss)
            loss = loss + args.branch_loss_weight * auxiliary_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_size = len(labels)
            loss_sum += float(loss.detach()) * batch_size
            classification_sum += float(terms["classification"]) * batch_size
            kl_sum += float(terms["kl"]) * batch_size
            auxiliary_sum += float(auxiliary_loss.detach()) * batch_size
            example_count += batch_size

        validation = evaluate(model, validation_loader, device)
        scheduler.step(validation["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / example_count,
            "classification_loss": classification_sum / example_count,
            "kl_loss": kl_sum / example_count,
            "branch_auxiliary_loss": auxiliary_sum / example_count,
            "annealing": annealing,
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "validation_uncertainty": validation["mean_uncertainty"],
            "validation_level_weight": validation["mean_level_weight"],
            "validation_trend_weight": validation["mean_trend_weight"],
        }
        history.append(row)
        print(
            f"[run {run_index + 1:02d}] epoch {epoch:02d}/{args.epochs} "
            f"loss={row['train_loss']:.5f} "
            f"val_accuracy={validation['accuracy']:.4f} "
            f"val_macro_F1={validation['macro_f1']:.4f} "
            f"val_uncertainty={validation['mean_uncertainty']:.4f} "
            f"weights=({validation['mean_level_weight']:.3f},"
            f"{validation['mean_trend_weight']:.3f}) "
            f"time={time.time() - started:.1f}s",
            flush=True,
        )
        if validation["macro_f1"] > best_f1 + 1e-6:
            best_f1 = validation["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[run {run_index + 1:02d}] early stopping", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    evaluation = evaluate(model, test_loader, device)
    metrics = make_metrics(evaluation, datasets["test"].class_names)
    run_dir = args.output / f"run_{run_index + 1:02d}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "checkpoint.pth")
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        actual=evaluation["actual"],
        predicted=evaluation["predicted"],
        probabilities=evaluation["probabilities"],
        uncertainty=evaluation["uncertainties"],
        level_uncertainty=evaluation["level_uncertainties"],
        trend_uncertainty=evaluation["trend_uncertainties"],
        level_weight=evaluation["level_weights"],
        trend_weight=evaluation["trend_weights"],
    )
    print(
        f"[run {run_index + 1:02d}] TEST accuracy={metrics['accuracy']:.4f} "
        f"macro_F1={metrics['macro_f1']:.4f} "
        f"weighted_F1={metrics['weighted_f1']:.4f} "
        f"uncertainty={metrics['mean_uncertainty']:.4f}"
        f" weights=({metrics['mean_level_weight']:.3f},"
        f"{metrics['mean_trend_weight']:.3f})"
    )
    for name, values in metrics["per_class"].items():
        print(
            f"  {name:<10} F1={values['f1']:.4f} recall={values['recall']:.4f} "
            f"precision={values['precision']:.4f} "
            f"uncertainty={values['mean_uncertainty']:.4f} "
            f"weights=({values['level_weight']:.3f},{values['trend_weight']:.3f}) "
            f"support={values['support']}"
        )
    print("Confusion matrix:")
    print(np.asarray(metrics["confusion_matrix"]))
    return metrics


def summarize_runs(run_metrics, class_names):
    summary = {}
    overall_metrics = (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "brier_score",
        "ece",
        "mean_uncertainty",
        "mean_level_weight",
        "mean_trend_weight",
    )
    for metric in overall_metrics:
        values = np.asarray([item[metric] for item in run_metrics])
        summary[metric] = {"mean": float(values.mean()), "sd": float(values.std())}
    summary["per_class"] = {}
    for name in class_names:
        summary["per_class"][name] = {}
        for metric in (
            "precision",
            "recall",
            "f1",
            "mean_uncertainty",
            "level_weight",
            "trend_weight",
        ):
            values = np.asarray(
                [item["per_class"][name][metric] for item in run_metrics]
            )
            summary["per_class"][name][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std()),
            }
    return summary


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    device = resolve_device(args.device)
    datasets = build_datasets(args)
    args.output.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output / "args.json").write_text(
        json.dumps(serializable_args, indent=2), encoding="utf-8"
    )
    for split, dataset in datasets.items():
        counts = ", ".join(
            f"{name}={int(count)}"
            for name, count in zip(dataset.class_names, dataset.class_counts)
        )
        print(f"{split}: {len(dataset)} windows ({counts})")
    probe = FireSemanticEvidenceLite(
        seq_len=args.seq_len,
        channels=len(datasets["train"].feature_names),
        embedding_dim=args.embedding_dim,
        evidence_hidden_dim=args.evidence_hidden_dim,
        num_classes=len(datasets["train"].class_names),
        dropout=args.dropout,
    )
    parameters = sum(parameter.numel() for parameter in probe.parameters())
    print(
        f"device={device} trainable_parameters={parameters:,} "
        "modules=level+trend+semantic_evidence persistence_gate=disabled"
    )

    results = [
        train_one_run(args, datasets, device, run_index)
        for run_index in range(args.runs)
    ]
    summary = summarize_runs(results, datasets["train"].class_names)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nFINAL TEST RESULTS (mean +/- SD)")
    for metric in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "mean_uncertainty",
        "mean_level_weight",
        "mean_trend_weight",
    ):
        values = summary[metric]
        print(f"{metric:<18} {values['mean']:.4f} +/- {values['sd']:.4f}")
    print("\nPER-CLASS TEST RESULTS (mean +/- SD)")
    for name, class_metrics in summary["per_class"].items():
        pieces = [
            f"{metric}={values['mean']:.4f}+/-{values['sd']:.4f}"
            for metric, values in class_metrics.items()
        ]
        print(f"{name:<10} " + " ".join(pieces))
    print(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()
