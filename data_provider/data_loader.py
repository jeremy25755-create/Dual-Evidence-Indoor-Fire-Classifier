import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.augmentation import run_augmentation_single
from utils.timefeatures import time_features

FIRE_SOURCE_CLASSES = ("Cable", "Candles", "Lunts", "Wood")
FIRE_STATE_CLASSES = ("Background", "Fire", "Nuisance")
FIRE_FEATURE_COLUMNS = (
    "CO2_Room",
    "CO_Room",
    "H2_Room",
    "Humidity_Room",
    "PM05_Room",
    "PM100_Room",
    "PM10_Room",
    "PM25_Room",
    "PM40_Room",
    "PM_Room_Typical_Size",
    "PM_Total_Room",
    "Temperature_Room",
    "UV_Room",
    "VOC_Room_RAW",
)


_FIRE_SOURCE_CACHE = {}
_FIRE_TERNARY_CACHE = {}


def _prepare_fire_ternary_data(
    csv_path,
    seq_len,
    stride,
    max_gap_seconds,
):
    """Create day-disjoint chronological windows for three fire states."""
    cache_key = (
        str(Path(csv_path).resolve()),
        int(seq_len),
        int(stride),
        float(max_gap_seconds),
    )
    if cache_key in _FIRE_TERNARY_CACHE:
        return _FIRE_TERNARY_CACHE[cache_key]

    required_columns = [
        "Date",
        "Sensor_ID",
        *FIRE_FEATURE_COLUMNS,
        "ternary_label",
    ]
    frame = pd.read_csv(csv_path, usecols=required_columns)
    frame = frame[frame["ternary_label"].isin(FIRE_STATE_CLASSES)].copy()
    frame["Date"] = pd.to_datetime(
        frame["Date"], format="mixed", utc=True, errors="raise"
    )
    frame.sort_values("Date", inplace=True)
    frame["day"] = frame["Date"].dt.strftime("%Y-%m-%d")
    days = sorted(frame["day"].unique().tolist())
    if len(days) < 3:
        raise ValueError(
            "Indoor fire ternary classification needs at least three dates "
            "for chronological train/validation/test splits"
        )
    split_days = {
        "train": days[:-2],
        "val": [days[-2]],
        "test": [days[-1]],
    }
    for split_name, assigned_days in split_days.items():
        labels = set(frame.loc[frame["day"].isin(assigned_days), "ternary_label"])
        missing = set(FIRE_STATE_CLASSES) - labels
        if missing:
            raise ValueError(
                f"{split_name} dates {assigned_days} are missing classes: {sorted(missing)}"
            )

    feature_columns = list(FIRE_FEATURE_COLUMNS)
    train_mask = frame["day"].isin(split_days["train"])
    scaler = StandardScaler().fit(frame.loc[train_mask, feature_columns])
    frame.loc[:, feature_columns] = scaler.transform(frame[feature_columns])
    state_to_index = {
        name: index for index, name in enumerate(FIRE_STATE_CLASSES)
    }

    prepared = {}
    for split_name, assigned_days in split_days.items():
        split_frame = frame[frame["day"].isin(assigned_days)]
        segments = []
        window_index = []
        labels = []
        for _, group in split_frame.groupby("Sensor_ID", sort=False):
            group = group.sort_values("Date")
            gaps = group["Date"].diff().dt.total_seconds().fillna(0).to_numpy()
            block_ids = np.cumsum(gaps > max_gap_seconds)
            for _, block in group.groupby(block_ids, sort=False):
                values = block.loc[:, feature_columns].to_numpy(
                    dtype=np.float32, copy=True
                )
                block_labels = (
                    block["ternary_label"].map(state_to_index).to_numpy(dtype=np.int64)
                )
                if len(values) < seq_len:
                    continue
                segment_index = len(segments)
                segments.append(values)
                for start in range(0, len(values) - seq_len + 1, stride):
                    window_index.append((segment_index, start))
                    labels.append(int(block_labels[start + seq_len - 1]))
        if not window_index:
            raise ValueError(
                f"No {split_name} windows were produced; check gap and length settings"
            )
        labels_array = np.asarray(labels, dtype=np.int64)
        if np.any(np.bincount(labels_array, minlength=3) == 0):
            raise ValueError(f"{split_name} windows do not contain all three classes")
        prepared[split_name] = {
            "segments": segments,
            "window_index": window_index,
            "labels": labels_array,
        }

    prepared["scaler"] = scaler
    prepared["split_days"] = split_days
    _FIRE_TERNARY_CACHE[cache_key] = prepared
    return prepared


class Dataset_IndoorFireTernary(Dataset):
    """Background/Fire/Nuisance windows split by whole calendar days."""

    def __init__(self, args, root_path, flag="train", size=None, **_):
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {flag}")
        seq_len = int(args.seq_len if size is None else size[0])
        prepared = _prepare_fire_ternary_data(
            csv_path=Path(root_path) / args.data_path,
            seq_len=seq_len,
            stride=int(args.classification_stride),
            max_gap_seconds=float(args.max_gap_seconds),
        )
        split = prepared[flag]
        self.seq_len = seq_len
        self.segments = split["segments"]
        self.window_index = split["window_index"]
        self.labels = split["labels"]
        self.scaler = prepared["scaler"]
        self.split_days = prepared["split_days"]
        self.class_names = FIRE_STATE_CLASSES
        self.feature_names = FIRE_FEATURE_COLUMNS
        self.class_counts = np.bincount(self.labels, minlength=len(self.class_names))

    def __getitem__(self, index):
        segment_index, start = self.window_index[index]
        window = self.segments[segment_index][start : start + self.seq_len]
        return torch.from_numpy(window), torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self):
        return len(self.window_index)


def _event_split(events):
    """Chronologically assign complete fire experiments to train/val/test."""
    if len(events) < 3:
        raise ValueError(
            "Each fire source needs at least three temporally separated experiments "
            "for leakage-free train/validation/test splits."
        )
    test_count = max(1, int(round(len(events) * 0.2)))
    validation_count = max(1, int(round(len(events) * 0.2)))
    if test_count + validation_count >= len(events):
        test_count = 1
        validation_count = 1
    train_end = len(events) - validation_count - test_count
    validation_end = len(events) - test_count
    return {
        "train": events[:train_end],
        "val": events[train_end:validation_end],
        "test": events[validation_end:],
    }


def _prepare_fire_source_data(
    csv_path,
    seq_len,
    stride,
    event_gap_seconds,
    max_gap_seconds,
    class_names,
):
    class_names = tuple(class_names)
    cache_key = (
        str(Path(csv_path).resolve()),
        int(seq_len),
        int(stride),
        float(event_gap_seconds),
        float(max_gap_seconds),
        class_names,
    )
    if cache_key in _FIRE_SOURCE_CACHE:
        return _FIRE_SOURCE_CACHE[cache_key]

    required_columns = [
        "Date",
        "Sensor_ID",
        *FIRE_FEATURE_COLUMNS,
        "scenario_label",
        "ternary_label",
    ]
    frame = pd.read_csv(csv_path, usecols=required_columns)
    frame = frame[
        frame["ternary_label"].eq("Fire")
        & frame["scenario_label"].isin(class_names)
    ].copy()
    if frame.empty:
        raise ValueError("No Fire-labelled rows with the four expected sources were found")

    frame["Date"] = pd.to_datetime(
        frame["Date"], format="mixed", utc=True, errors="raise"
    )
    frame.sort_values("Date", inplace=True)
    frame["event_key"] = ""
    split_events = {"train": [], "val": [], "test": []}
    event_summary = {
        name: {"train": [], "val": [], "test": []} for name in class_names
    }
    source_to_index = {name: index for index, name in enumerate(class_names)}

    for source in class_names:
        source_index = frame.index[frame["scenario_label"].eq(source)]
        source_times = frame.loc[source_index, "Date"]
        local_event = (
            source_times.diff().dt.total_seconds().fillna(event_gap_seconds + 1)
            > event_gap_seconds
        ).cumsum() - 1
        keys = source + "_" + local_event.astype(str)
        frame.loc[source_index, "event_key"] = keys.values
        ordered_events = (
            frame.loc[source_index]
            .groupby("event_key", sort=False)["Date"]
            .min()
            .sort_values()
            .index.tolist()
        )
        source_split = _event_split(ordered_events)
        for split_name, assigned in source_split.items():
            split_events[split_name].extend(assigned)
            event_summary[source][split_name] = assigned

    feature_columns = list(FIRE_FEATURE_COLUMNS)
    train_mask = frame["event_key"].isin(split_events["train"])
    scaler = StandardScaler().fit(frame.loc[train_mask, feature_columns])
    frame.loc[:, feature_columns] = scaler.transform(frame[feature_columns])

    prepared = {}
    for split_name, assigned_events in split_events.items():
        split_frame = frame[frame["event_key"].isin(assigned_events)]
        segments = []
        window_index = []
        labels = []
        window_events = []
        for (event_key, sensor_id), group in split_frame.groupby(
            ["event_key", "Sensor_ID"], sort=False
        ):
            group = group.sort_values("Date")
            gaps = group["Date"].diff().dt.total_seconds().fillna(0).to_numpy()
            block_ids = np.cumsum(gaps > max_gap_seconds)
            source_name = str(group["scenario_label"].iloc[0])
            label = source_to_index[source_name]
            for _, block in group.groupby(block_ids, sort=False):
                values = block.loc[:, feature_columns].to_numpy(
                    dtype=np.float32, copy=True
                )
                if len(values) < seq_len:
                    continue
                segment_index = len(segments)
                segments.append(values)
                for start in range(0, len(values) - seq_len + 1, stride):
                    window_index.append((segment_index, start))
                    labels.append(label)
                    window_events.append(str(event_key))
        if not window_index:
            raise ValueError(f"No {split_name} windows were produced; check gap and length settings")
        prepared[split_name] = {
            "segments": segments,
            "window_index": window_index,
            "labels": np.asarray(labels, dtype=np.int64),
            "events": tuple(window_events),
        }

    prepared["scaler"] = scaler
    prepared["event_summary"] = event_summary
    prepared["class_names"] = class_names
    _FIRE_SOURCE_CACHE[cache_key] = prepared
    return prepared


class Dataset_IndoorFireSource(Dataset):
    """Fire-only, event-disjoint classification windows for four fire sources."""

    def __init__(self, args, root_path, flag="train", size=None, **_):
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {flag}")
        if size is None:
            seq_len = int(args.seq_len)
        else:
            seq_len = int(size[0])
        class_names = (
            FIRE_SOURCE_CLASSES[:-1]
            if bool(getattr(args, "exclude_wood", False))
            else FIRE_SOURCE_CLASSES
        )
        csv_path = Path(root_path) / args.data_path
        prepared = _prepare_fire_source_data(
            csv_path=csv_path,
            seq_len=seq_len,
            stride=int(args.classification_stride),
            event_gap_seconds=float(args.event_gap_seconds),
            max_gap_seconds=float(args.max_gap_seconds),
            class_names=class_names,
        )
        split = prepared[flag]
        self.seq_len = seq_len
        self.segments = split["segments"]
        self.window_index = split["window_index"]
        self.labels = split["labels"]
        self.event_keys = split["events"]
        self.event_summary = prepared["event_summary"]
        self.scaler = prepared["scaler"]
        self.class_names = prepared["class_names"]
        self.feature_names = FIRE_FEATURE_COLUMNS
        self.class_counts = np.bincount(
            self.labels, minlength=len(self.class_names)
        )

    def __getitem__(self, index):
        segment_index, start = self.window_index[index]
        window = self.segments[segment_index][start : start + self.seq_len]
        return torch.from_numpy(window), torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self):
        return len(self.window_index)


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
