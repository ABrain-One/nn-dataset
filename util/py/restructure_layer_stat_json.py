import argparse
import json
from pathlib import Path

from ab.nn.util.Const import stat_train_dir

# This is the layer-analysis schedule used by Train.py:
# epochs 1-5, then every fifth epoch.
def _is_analysis_epoch(epoch: int) -> bool:
    return epoch >= 1 and (epoch <= 5 or epoch % 5 == 0)

def _expected_snapshot_count(epoch: int) -> int:
    """Number of cumulative layer snapshots expected by this epoch."""
    return sum(
        1
        for candidate in range(1, epoch + 1)
        if _is_analysis_epoch(candidate)
    )

def _is_valid_snapshot(value) -> bool:
    """Return whether value has the object shape expected by Write.py."""
    return isinstance(value, dict) and isinstance(value.get("layers"), list)

def restructure_record(record: dict, epoch: int) -> tuple[bool, str, str]:
    """
    Convert or remove layer_stat in one training record.

    Returns (changed, action, detail), where action is one of:
      - unchanged
      - converted_legacy
      - removed_non_analysis
      - removed_ambiguous_legacy
      - removed_malformed
    """
    if not isinstance(record, dict) or "layer_stat" not in record:
        return False, "unchanged", ""

    layer_stat = record.get("layer_stat")
    if layer_stat is None:
        return False, "unchanged", ""

    # Any snapshot stored in a non-analysis epoch is stale/unintended.
    if not _is_analysis_epoch(epoch):
        del record["layer_stat"]
        return True, "removed_non_analysis", f"epoch {epoch} is not an analysis epoch"

    # Current format: preserve one valid snapshot object.
    if isinstance(layer_stat, dict):
        if _is_valid_snapshot(layer_stat):
            return False, "unchanged", "current object format"

        del record["layer_stat"]
        return True, "removed_malformed", "object has no valid layers list"

    # Legacy format: a cumulative list of snapshots.
    if isinstance(layer_stat, list):
        expected = _expected_snapshot_count(epoch)

        if (
            len(layer_stat) == expected
            and layer_stat
            and _is_valid_snapshot(layer_stat[-1])
        ):
            record["layer_stat"] = layer_stat[-1]

            return (
                True,
                "converted_legacy",
                f"{len(layer_stat)} cumulative snapshots -> epoch {epoch} snapshot",
            )

        # There is no epoch identifier inside the old list. If its shape does
        # not prove which snapshot belongs to this file, remove it rather than
        # assigning a stale snapshot to the wrong epoch.
        del record["layer_stat"]

        return (
            True,
            "removed_ambiguous_legacy",
            f"found {len(layer_stat)} snapshots, expected {expected}",
        )

    # The importer cannot use any other representation safely.
    del record["layer_stat"]
    return True, "removed_malformed", f"unexpected type {type(layer_stat).__name__}"

def process_file(
    epoch_file: Path,
    write: bool = False,
) -> tuple[bool, int, dict[str, int]]:
    with open(epoch_file, "r", encoding="utf-8") as handle:
        trials = json.load(handle)

    if not isinstance(trials, list):
        return False, 0, {}

    try:
        epoch = int(epoch_file.stem)
    except ValueError:
        return False, 0, {}

    changed_records = 0
    action_counts: dict[str, int] = {}
    details: list[str] = []

    for index, record in enumerate(trials):
        changed, action, detail = restructure_record(record, epoch)

        if not changed:
            continue

        changed_records += 1
        action_counts[action] = action_counts.get(action, 0) + 1
        details.append(f"trial {index}: {action} ({detail})")

    if changed_records and write:
        with open(epoch_file, "w", encoding="utf-8") as handle:
            json.dump(trials, handle, indent=4, ensure_ascii=False)
            handle.write("\n")

    if changed_records:
        mode = "Updated" if write else "Would update"
        print(f"{mode}: {epoch_file}")

        for detail in details:
            print(f"  - {detail}")

    return changed_records > 0, changed_records, action_counts

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert cumulative legacy layer_stat snapshots into one snapshot "
            "per intended analysis epoch."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite JSON files. Without this flag, only a dry run is performed.",
    )

    args = parser.parse_args()

    if not stat_train_dir.exists():
        raise FileNotFoundError(
            f"Training-stat directory not found: {stat_train_dir}"
        )

    totals = {
        "files": 0,
        "records": 0,
        "converted_legacy": 0,
        "removed_non_analysis": 0,
        "removed_ambiguous_legacy": 0,
        "removed_malformed": 0,
        "invalid_files": 0,
    }

    print(f"Scanning: {stat_train_dir}")
    print("Mode:", "WRITE" if args.write else "DRY RUN")
    print()

    for config_dir in sorted(
        stat_train_dir.iterdir(),
        key=lambda path: path.name,
    ):
        if not config_dir.is_dir():
            continue

        for epoch_file in sorted(
            config_dir.iterdir(),
            key=lambda path: path.name,
        ):
            if (
                not epoch_file.is_file()
                or epoch_file.suffix.lower() != ".json"
            ):
                continue

            try:
                int(epoch_file.stem)
            except ValueError:
                continue

            try:
                changed, changed_records, action_counts = process_file(
                    epoch_file,
                    write=args.write,
                )
            except Exception as exc:
                totals["invalid_files"] += 1
                print(f"ERROR: {epoch_file} -> {exc}")
                continue

            if changed:
                totals["files"] += 1
                totals["records"] += changed_records

                for action, count in action_counts.items():
                    totals[action] += count

    print()
    print("Summary")
    print("-------")
    print(f"Changed files:                  {totals['files']}")
    print(f"Changed records:                {totals['records']}")
    print(f"Legacy lists converted:         {totals['converted_legacy']}")
    print(f"Non-analysis snapshots removed: {totals['removed_non_analysis']}")
    print(
        "Ambiguous legacy lists removed: "
        f"{totals['removed_ambiguous_legacy']}"
    )
    print(f"Malformed values removed:       {totals['removed_malformed']}")
    print(f"Invalid files:                  {totals['invalid_files']}")

    if not args.write:
        print()
        print("Dry run only. Re-run with --write to update JSON files.")

if __name__ == "__main__":
    main()