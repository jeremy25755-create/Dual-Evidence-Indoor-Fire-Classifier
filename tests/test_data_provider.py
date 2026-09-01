import math
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from data_provider.data_factory import data_provider
from data_provider.data_loader import (
    FIRE_FEATURE_COLUMNS,
    FIRE_SOURCE_CLASSES,
    FIRE_STATE_CLASSES,
)


def make_args(root_path):
    return SimpleNamespace(
        data="custom",
        embed="timeF",
        root_path=str(root_path),
        data_path="sample.csv",
        seq_len=24,
        label_len=12,
        pred_len=12,
        features="M",
        target="OT",
        freq="h",
        seasonal_patterns="Monthly",
        augmentation_ratio=0,
        batch_size=16,
        num_workers=0,
        prefetch_factor=2,
        use_gpu=False,
        seed=2021,
        task_name="long_term_forecast",
    )


def write_dataset(path):
    rows = 240
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "feature": range(rows),
            "OT": range(rows, 2 * rows),
        }
    )
    frame.to_csv(path / "sample.csv", index=False)


def test_train_loader_is_seeded_shuffled_and_drops_partial_batch(tmp_path):
    write_dataset(tmp_path)
    _, loader = data_provider(make_args(tmp_path), "train")

    assert isinstance(loader.sampler, RandomSampler)
    assert loader.drop_last is True
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))
    assert batch_x.shape == (16, 24, 2)
    assert batch_y.shape == (16, 24, 2)
    assert batch_x_mark.shape[1] == 24
    assert batch_y_mark.shape[1] == 24
    assert batch_x.dtype == torch.float64


def test_evaluation_loader_is_ordered_and_keeps_partial_batch(tmp_path):
    write_dataset(tmp_path)
    _, loader = data_provider(make_args(tmp_path), "test")

    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.drop_last is False
    expected_batches = math.ceil(len(loader.dataset) / loader.batch_size)
    assert len(list(loader)) == expected_batches


def write_fire_dataset(path):
    rows = []
    base = pd.Timestamp("2022-07-01", tz="UTC")
    for source_index, source in enumerate(FIRE_SOURCE_CLASSES):
        for event_index in range(3):
            event_start = base + pd.Timedelta(hours=12 * event_index + source_index * 2)
            for step in range(8):
                for sensor_index, sensor in enumerate(("SensorA", "SensorB")):
                    row = {
                        "Date": event_start
                        + pd.Timedelta(seconds=10 * step, milliseconds=100 * sensor_index),
                        "Sensor_ID": sensor,
                        "scenario_label": source,
                        "ternary_label": "Fire",
                    }
                    for feature_index, feature in enumerate(FIRE_FEATURE_COLUMNS):
                        row[feature] = source_index + event_index + step + feature_index / 10
                    rows.append(row)
    pd.DataFrame(rows).to_csv(path / "fire.csv", index=False)


def fire_args(root_path):
    return SimpleNamespace(
        data="IndoorFireSource",
        embed="timeF",
        root_path=str(root_path),
        data_path="fire.csv",
        seq_len=4,
        label_len=0,
        pred_len=0,
        features="M",
        target="unused",
        freq="s",
        seasonal_patterns="Monthly",
        augmentation_ratio=0,
        batch_size=8,
        num_workers=0,
        prefetch_factor=2,
        use_gpu=False,
        seed=2021,
        task_name="classification",
        classification_stride=2,
        event_gap_seconds=60,
        max_gap_seconds=20,
        exclude_wood=False,
    )


def test_fire_source_split_is_event_disjoint_and_contains_every_class(tmp_path):
    write_fire_dataset(tmp_path)
    train, train_loader = data_provider(fire_args(tmp_path), "train")
    validation, _ = data_provider(fire_args(tmp_path), "val")
    test, _ = data_provider(fire_args(tmp_path), "test")

    assert train.class_counts.tolist() == [6, 6, 6, 6]
    assert validation.class_counts.tolist() == [6, 6, 6, 6]
    assert test.class_counts.tolist() == [6, 6, 6, 6]
    assert set(train.event_keys).isdisjoint(validation.event_keys)
    assert set(train.event_keys).isdisjoint(test.event_keys)
    assert set(validation.event_keys).isdisjoint(test.event_keys)
    windows, labels = next(iter(train_loader))
    assert windows.shape[1:] == (4, len(FIRE_FEATURE_COLUMNS))
    assert labels.dtype == torch.long


def test_fire_source_can_exclude_wood(tmp_path):
    write_fire_dataset(tmp_path)
    args = fire_args(tmp_path)
    args.exclude_wood = True

    train, _ = data_provider(args, "train")
    validation, _ = data_provider(args, "val")
    test, _ = data_provider(args, "test")

    assert train.class_names == ("Cable", "Candles", "Lunts")
    assert train.class_counts.tolist() == [6, 6, 6]
    assert validation.class_counts.tolist() == [6, 6, 6]
    assert test.class_counts.tolist() == [6, 6, 6]
    assert set(train.event_keys).isdisjoint(validation.event_keys)
    assert set(train.event_keys).isdisjoint(test.event_keys)


def write_ternary_dataset(path):
    rows = []
    base = pd.Timestamp("2022-07-04", tz="UTC")
    for day_index in range(5):
        day_start = base + pd.Timedelta(days=day_index)
        for state_index, state in enumerate(FIRE_STATE_CLASSES):
            block_start = day_start + pd.Timedelta(minutes=2 * state_index)
            for step in range(8):
                for sensor_index, sensor in enumerate(("SensorA", "SensorB")):
                    row = {
                        "Date": block_start
                        + pd.Timedelta(seconds=10 * step, milliseconds=100 * sensor_index),
                        "Sensor_ID": sensor,
                        "ternary_label": state,
                    }
                    for feature_index, feature in enumerate(FIRE_FEATURE_COLUMNS):
                        row[feature] = day_index + state_index + feature_index / 10
                    rows.append(row)
    pd.DataFrame(rows).to_csv(path / "ternary.csv", index=False)


def test_fire_ternary_split_uses_disjoint_chronological_days(tmp_path):
    write_ternary_dataset(tmp_path)
    args = fire_args(tmp_path)
    args.data = "IndoorFireTernary"
    args.data_path = "ternary.csv"

    train, _ = data_provider(args, "train")
    validation, _ = data_provider(args, "val")
    test, _ = data_provider(args, "test")

    assert train.class_names == ("Background", "Fire", "Nuisance")
    assert train.class_counts.tolist() == [18, 18, 18]
    assert validation.class_counts.tolist() == [6, 6, 6]
    assert test.class_counts.tolist() == [6, 6, 6]
    assert train.split_days == {
        "train": ["2022-07-04", "2022-07-05", "2022-07-06"],
        "val": ["2022-07-07"],
        "test": ["2022-07-08"],
    }
