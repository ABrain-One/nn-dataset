import argparse
import json
import os.path
import shutil

from ab.nn.util.Const import stat_train_dir


TRAIN_STAT_FIELDS = (
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
)


def restructure_record(prm):
    """
    Move old top-level training statistics into nested train_stat.

    Example:
        train_loss -> train_stat["train_loss"]

    Existing train_stat values are preserved unless they are missing or None.
    Old top-level fields are removed after being moved.
    """
    if not isinstance(prm, dict):
        return False, 0

    train_stat = prm.get("train_stat")

    if isinstance(train_stat, dict):
        train_stat = dict(train_stat)
    else:
        train_stat = {}

    changed = False
    moved_count = 0

    for field in TRAIN_STAT_FIELDS:
        if field in prm:
            if field not in train_stat or train_stat[field] is None:
                train_stat[field] = prm[field]

            del prm[field]
            changed = True
            moved_count += 1

    if train_stat:
        if prm.get("train_stat") != train_stat:
            prm["train_stat"] = train_stat
            changed = True

    if isinstance(prm.get("train_stat"), dict) and "epoch_max" in prm["train_stat"]:
        if "epoch_max" not in prm or prm["epoch_max"] is None:
            prm["epoch_max"] = prm["train_stat"]["epoch_max"]

        del prm["train_stat"]["epoch_max"]
        changed = True
        moved_count += 1

        if not prm["train_stat"]:
            del prm["train_stat"]

    return changed, moved_count


def process_file(epoch_file, write=False, backup=False):
    with open(epoch_file, "r", encoding="utf-8") as f:
        trials = json.load(f)

    if not isinstance(trials, list):
        return False, 0, 0

    changed_records = 0
    moved_values = 0

    for prm in trials:
        changed, moved_count = restructure_record(prm)

        if changed:
            changed_records += 1
            moved_values += moved_count

    if changed_records > 0 and write:
        if backup:
            backup_file = str(epoch_file) + ".bak"
            shutil.copy2(epoch_file, backup_file)

        with open(epoch_file, "w", encoding="utf-8") as f:
            json.dump(trials, f, indent=4, ensure_ascii=False)
            f.write("\n")

    return changed_records > 0, changed_records, moved_values


def main():
    parser = argparse.ArgumentParser(
        description="Restructure old training JSON files by moving training statistics into nested train_stat."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite JSON files. Without this flag, only a dry run is performed.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backup files before rewriting JSON files.",
    )

    args = parser.parse_args()

    changed_files = 0
    changed_records_total = 0
    moved_values_total = 0

    print(f"Scanning: {stat_train_dir}")
    print("Mode:", "WRITE" if args.write else "DRY RUN")
    print()

    for p in stat_train_dir.iterdir():
        if not os.path.isdir(p):
            continue

        for epoch_file in p.iterdir():
            try:
                int(epoch_file.stem)
            except Exception:
                continue

            if epoch_file.suffix.lower() != ".json":
                continue

            try:
                changed, changed_records, moved_values = process_file(
                    epoch_file,
                    write=args.write,
                    backup=args.backup,
                )
            except Exception as e:
                print(f"ERROR: {epoch_file} -> {e}")
                continue

            if changed:
                changed_files += 1
                changed_records_total += changed_records
                moved_values_total += moved_values
                print(
                    f"Updated: {epoch_file} | "
                    f"records changed: {changed_records} | "
                    f"values moved: {moved_values}"
                )

    print()
    print("Summary")
    print("-------")
    print(f"Changed files: {changed_files}")
    print(f"Changed records: {changed_records_total}")
    print(f"Moved values: {moved_values_total}")

    if not args.write:
        print()
        print("Dry run only. Re-run with --write to update files.")


if __name__ == "__main__":
    main()