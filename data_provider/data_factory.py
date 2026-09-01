"""Dataset and DataLoader construction."""

import torch
from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_IndoorFireSource,
    Dataset_IndoorFireTernary,
)

DATASETS = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "IndoorFireSource": Dataset_IndoorFireSource,
    "IndoorFireTernary": Dataset_IndoorFireTernary,
}


def data_provider(args, flag):
    if flag not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported data split: {flag}")
    if args.data not in DATASETS:
        raise ValueError(
            f"Unknown dataset {args.data!r}. Choose from: {', '.join(DATASETS)}"
        )

    dataset_class = DATASETS[args.data]
    time_encoding = 0 if args.embed != "timeF" else 1
    is_training = flag == "train"
    worker_count = max(0, int(args.num_workers))

    dataset = dataset_class(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=time_encoding,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns,
    )
    print(flag, len(dataset))
    if hasattr(dataset, "class_counts"):
        counts = ", ".join(
            f"{name}={int(count)}"
            for name, count in zip(dataset.class_names, dataset.class_counts)
        )
        print(f"{flag} class windows: {counts}")

    generator = torch.Generator()
    generator.manual_seed(getattr(args, "seed", 2021))
    loader_options = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": is_training,
        "num_workers": worker_count,
        "drop_last": is_training and args.task_name != "classification",
        "pin_memory": bool(args.use_gpu and torch.cuda.is_available()),
        "generator": generator,
    }
    if worker_count > 0:
        loader_options["prefetch_factor"] = max(1, int(args.prefetch_factor))
        loader_options["persistent_workers"] = True

    return dataset, DataLoader(**loader_options)
