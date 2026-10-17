#!/usr/bin/env python3
"""Normalize LR scheduler statistics for the NN Dataset importer.

This utility is intentionally scoped to an explicit statistics directory. It
does not scan or rewrite the repository's existing student statistics.
"""

import argparse
import json
from pathlib import Path


TRAIN_STAT_FIELDS = {
    "train_loss",
    "test_loss",
    "train_accuracy",
    "gradient_norm",
    "samples_per_second",
    "best_accuracy",
    "best_epoch",
    "cpu_count",
    "cpu_type",
    "cpu_usage_percent",
    "total_ram_kb",
    "occupied_ram_kb",
    "ram_usage_percent",
    "gpu_type",
    "gpu_memory_kb",
    "gpu_total_memory_kb",
    "occupied_gpu_memory_kb",
    "gpu_memory_usage_percent",
}

REMOVED_FIELDS = {
    "architecture",
    "code_hash",
    "epoch",
    "epoch_metrics",
    "hyperparameters",
    "model_id",
    "scheduler",
    "source_id",
    "statistics_source",
}


def normalize_record(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("every trial record must be an object")

    normalized = dict(record)
    nested_hyperparameters = normalized.pop("hyperparameters", {})
    nested_epoch_metrics = normalized.pop("epoch_metrics", {})

    if nested_hyperparameters and not isinstance(nested_hyperparameters, dict):
        raise ValueError("hyperparameters must be an object")
    if nested_epoch_metrics and not isinstance(nested_epoch_metrics, dict):
        raise ValueError("epoch_metrics must be an object")

    # Nested values are defaults; explicit top-level values remain authoritative.
    flattened = dict(nested_hyperparameters)
    for name, value in nested_epoch_metrics.items():
        flattened.setdefault(name, value)
    for name, value in flattened.items():
        normalized.setdefault(name, value)

    train_stat = normalized.get("train_stat")
    if train_stat is None:
        train_stat = {}
    elif not isinstance(train_stat, dict):
        raise ValueError("train_stat must be an object when present")
    else:
        train_stat = dict(train_stat)

    for field in TRAIN_STAT_FIELDS:
        if field in normalized:
            train_stat.setdefault(field, normalized.pop(field))

    if train_stat:
        normalized["train_stat"] = train_stat
    else:
        normalized.pop("train_stat", None)

    for field in REMOVED_FIELDS:
        normalized.pop(field, None)

    required = {"accuracy", "batch", "lr", "momentum", "transform", "uid"}
    missing = sorted(required - normalized.keys())
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    return normalized


def normalize_file(path: Path, write: bool) -> tuple[int, int]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("statistics file must contain a JSON array")

    normalized_records = [normalize_record(record) for record in records]
    changed = int(normalized_records != records)
    if write and changed:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(normalized_records, handle, indent=4, ensure_ascii=False)
            handle.write("\n")
    return changed, len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("statistics_root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    files = sorted(args.statistics_root.glob("img-classification_cifar-10_acc_lr-*/[0-9]*.json"))
    changed_files = 0
    record_count = 0
    for path in files:
        changed, records = normalize_file(path, args.write)
        changed_files += changed
        record_count += records

    mode = "wrote" if args.write else "would rewrite"
    print(f"{mode} {changed_files} files; validated {record_count} records")


if __name__ == "__main__":
    main()