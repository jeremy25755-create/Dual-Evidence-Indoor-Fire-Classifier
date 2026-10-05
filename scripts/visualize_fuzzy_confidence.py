"""Visualize learned fuzzy confidence on the held-out test split."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    "TEFN_FuzzyTCN_Classifier_32_"
    "FuzzyHybridDeltaTCN_L3_h64_tcn8_HybridEDL_w085_"
    "p96_e2_N1_T1_C1_add_R1_P0_D0.1_68722c"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "out" / "interpretability" / "fuzzy_confidence.png",
    )
    parser.add_argument("--no_show", action="store_true",
                        help="Save the figure without opening the result window")
    return parser.parse_args()


def load_run(run_name, device):
    sys.path.insert(0, str(ROOT))
    from data_provider.data_loader import Dataset_IndoorFireTernary
    from models.TEFN_FuzzyTCN_Classifier_32 import Model

    result_dir = ROOT / "out" / "results" / run_name
    with (result_dir / "args.json").open(encoding="utf-8") as file:
        config = SimpleNamespace(**json.load(file))

    data_root = Path(config.root_path)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    dataset = Dataset_IndoorFireTernary(config, root_path=data_root, flag="test")
    model = Model(config).to(device)
    checkpoint = torch.load(
        ROOT / "checkpoints" / run_name / "checkpoint.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint)
    model.eval()

    saved = np.load(result_dir / "predictions.npz")
    return model, dataset, saved


@torch.inference_mode()
def collect_confidence(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    confidence, labels = [], []
    for windows, targets in loader:
        membership = model.sensor.fuzzy(windows.to(device))
        entropy = -(membership * membership.clamp_min(1e-8).log()).sum(dim=-1)
        confidence.append((1.0 - entropy / math.log(model.sensor.levels)).cpu())
        labels.append(targets)
    return torch.cat(confidence).numpy(), torch.cat(labels).numpy()


def short_sensor_names(names):
    return [
        name.replace("_Room_Typical_Size", " typical size")
        .replace("_Room_RAW", "")
        .replace("_Room", "")
        .replace("PM_Total", "PM total")
        for name in names
    ]


def summarize(confidence, actual, predicted, class_names, sensor_names):
    window_conf = confidence.mean(axis=(1, 2))
    class_sensor = np.stack(
        [confidence[actual == index].mean(axis=(0, 1)) for index in range(len(class_names))],
        axis=1,
    )
    time_profile = np.stack(
        [confidence[actual == index].mean(axis=(0, 2)) for index in range(len(class_names))]
    )
    correct = actual == predicted
    summary = {
        "definition": "1 - normalized entropy of fuzzy memberships",
        "overall_mean": float(window_conf.mean()),
        "correct_mean": float(window_conf[correct].mean()),
        "incorrect_mean": float(window_conf[~correct].mean()),
        "class_mean": {
            name: float(window_conf[actual == index].mean())
            for index, name in enumerate(class_names)
        },
        "most_confident_sensor_by_class": {
            class_names[index]: {
                "sensor": sensor_names[int(class_sensor[:, index].argmax())],
                "confidence": float(class_sensor[:, index].max()),
            }
            for index in range(len(class_names))
        },
    }
    return window_conf, class_sensor, time_profile, correct, summary


def draw(output, window_conf, class_sensor, time_profile, correct, actual,
         class_names, sensor_names):
    colors = ("#3274A1", "#E1812C", "#3A923A")
    fig = plt.figure(figsize=(13.5, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.55, 1.0))

    ax = fig.add_subplot(grid[0, :])
    image = ax.imshow(class_sensor, aspect="auto", cmap="viridis", norm=Normalize(0, 1))
    ax.set_title("Learned fuzzy confidence by sensor and true class")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(sensor_names)), sensor_names)
    ax.set_xlabel("True class")
    ax.set_ylabel("Sensor")
    for row in range(class_sensor.shape[0]):
        for column in range(class_sensor.shape[1]):
            value = class_sensor[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    color="white" if value < 0.45 else "black", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, pad=0.01, shrink=0.9)
    colorbar.set_label("Mean fuzzy confidence (0–1)")

    ax = fig.add_subplot(grid[1, 0])
    bins = np.linspace(0, 1, 31)
    ax.hist(window_conf[correct], bins=bins, density=True, alpha=0.62,
            color=colors[0], label=f"Correct (n={correct.sum():,})")
    ax.hist(window_conf[~correct], bins=bins, density=True, alpha=0.62,
            color=colors[1], label=f"Incorrect (n={(~correct).sum():,})")
    ax.axvline(window_conf[correct].mean(), color=colors[0], linewidth=2)
    ax.axvline(window_conf[~correct].mean(), color=colors[1], linewidth=2)
    ax.set_title("Window-level confidence by prediction outcome")
    ax.set_xlabel("Mean fuzzy confidence")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[1, 1])
    steps = np.arange(1, time_profile.shape[1] + 1)
    for index, name in enumerate(class_names):
        ax.plot(steps, time_profile[index], color=colors[index], linewidth=2,
                marker="o", markevery=4, markersize=4, label=name)
    ax.set_title("Confidence across the 32-step input window")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean fuzzy confidence")
    ax.set_xlim(1, len(steps))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle("TEFN FuzzyTCN — Fuzzy Confidence on the Test Set", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, dataset, saved = load_run(args.run, device)
    confidence, labels = collect_confidence(model, dataset, device, args.batch_size)
    actual, predicted = saved["actual"], saved["predicted"]
    if not np.array_equal(labels, actual):
        raise RuntimeError("Saved predictions and reconstructed test windows are misaligned")

    class_names = list(dataset.class_names)
    sensor_names = short_sensor_names(dataset.feature_names)
    values = summarize(confidence, actual, predicted, class_names, sensor_names)
    window_conf, class_sensor, time_profile, correct, summary = values
    draw(args.output, window_conf, class_sensor, time_profile, correct, actual,
         class_names, sensor_names)

    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = args.output.with_name(args.output.stem + "_by_sensor.csv")
    header = "sensor," + ",".join(class_names)
    rows = [sensor_names[i] + "," + ",".join(f"{value:.6f}" for value in class_sensor[i])
            for i in range(len(sensor_names))]
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"figure={args.output}")
    print(f"summary={summary_path}")
    print(f"sensor_csv={csv_path}")
    if not args.no_show:
        os.startfile(args.output.resolve())


if __name__ == "__main__":
    main()
