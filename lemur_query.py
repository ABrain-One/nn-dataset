"""
lemur_query.py — Standalone User-Facing Extraction Tool for the LEMUR NN Dataset
==================================================================================

Usage examples:

  # List all available tasks
  python lemur_query.py

  # Show models + datasets + best accuracy for a specific task
  python lemur_query.py --task "image classification"

  # Export task results to Excel (.xlsx) or CSV
  python lemur_query.py --task "image classification" --csv results.xlsx
  python lemur_query.py --task "image classification" --csv results.csv

  # Export task results to plain text
  python lemur_query.py --task "image classification" --txt results.txt

  # Extract source code for one or more models by name prefix
  python lemur_query.py --code AlexNet
  python lemur_query.py --code AlexNet,ResNet --out-dir ./model_code

  # Best accuracy per model grouped by dataset (all tasks)
  python lemur_query.py --best-accuracy

  # Best accuracy per model grouped by dataset, filtered by task
  python lemur_query.py --best-accuracy --task "image classification"

  # Export best-accuracy table to CSV
  python lemur_query.py --best-accuracy --csv best.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Export helper — writes .xlsx or .csv depending on file extension
# ---------------------------------------------------------------------------
def _export_table(df, path: str, label: str) -> None:
    """Save df to Excel if path ends with .xlsx, otherwise CSV."""
    if path.lower().endswith(".xlsx"):
        df.to_excel(path, index=False, engine="openpyxl")
    else:
        df.to_csv(path, index=False)
    print(f"{label} saved -> {path}")


# ---------------------------------------------------------------------------
# Lazy DB load — only called when data is actually needed
# ---------------------------------------------------------------------------
def _load_df(task: str | None = None, only_best: bool = False):
    try:
        import ab.nn.api as api  # type: ignore
    except ImportError:
        print("ERROR: Could not import 'ab.nn.api'. "
              "Make sure you are running inside the nn-dataset environment.", file=sys.stderr)
        sys.exit(1)
    try:
        df = api.data(only_best_accuracy=only_best, task=task if task else None)
    except Exception as exc:
        print(f"ERROR: Failed to load dataset: {exc}", file=sys.stderr)
        sys.exit(1)
    if df.empty:
        print("No data found in the database.")
        sys.exit(0)
    return df


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------
def cmd_list_tasks() -> None:
    """Print all unique task names available in the database."""
    df = _load_df()
    if "task" not in df.columns:
        print("ERROR: 'task' column not found in the dataset.", file=sys.stderr)
        sys.exit(1)
    tasks = sorted(df["task"].dropna().unique())
    print(f"\nAvailable tasks ({len(tasks)}):\n")
    for t in tasks:
        print(f"  {t}")
    print()


def cmd_task_summary(task: str, csv_out: str | None, txt_out: str | None) -> None:
    """
    For a given task, show each model + dataset combination and the maximum
    accuracy achieved. Optionally export to CSV or TXT.
    """
    import pandas as pd

    df = _load_df(task=task)

    required = {"task", "nn", "dataset", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns in dataset: {missing}", file=sys.stderr)
        sys.exit(1)

    # Filter to exact task match (api may return all if task arg is broad)
    mask = df["task"].str.lower() == task.lower()
    df_task = df[mask]
    if df_task.empty:
        # Try partial match
        mask = df["task"].str.lower().str.contains(task.lower(), na=False)
        df_task = df[mask]
    if df_task.empty:
        print(f"No data found for task: {task!r}")
        return

    actual_task = df_task["task"].iloc[0]
    summary = (
        df_task.groupby(["nn", "dataset"], sort=True)["accuracy"]
        .max()
        .reset_index()
        .rename(columns={"nn": "Model", "dataset": "Dataset", "accuracy": "Max Accuracy"})
        .sort_values(["Dataset", "Max Accuracy"], ascending=[True, False])
    )
    summary["Max Accuracy"] = summary["Max Accuracy"].round(6)

    print(f"\nTask: {actual_task}")
    print(f"{'─' * 70}")
    print(summary.to_string(index=False))
    print()

    if csv_out:
        _export_table(summary, csv_out, "Table")

    if txt_out:
        with open(txt_out, "w", encoding="utf-8") as fh:
            fh.write(f"Task: {actual_task}\n")
            fh.write("─" * 70 + "\n")
            fh.write(summary.to_string(index=False))
            fh.write("\n")
        print(f"TXT saved -> {txt_out}")


def cmd_extract_code(prefixes: list[str], out_dir: str | None) -> None:
    """
    Retrieve and save the source code for all models whose name starts with
    any of the given prefixes. One .py file per unique model name.
    """
    import ab.nn.api as api  # type: ignore

    dest = Path(out_dir) if out_dir else Path(".")
    dest.mkdir(parents=True, exist_ok=True)

    found_any = False
    for prefix in prefixes:
        prefix = prefix.strip()
        print(f"\nSearching for models with prefix: {prefix!r} ...")
        try:
            df = api.data(nn_prefixes=(prefix,))
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue

        if df.empty or "nn" not in df.columns or "nn_code" not in df.columns:
            print(f"  No models found for prefix {prefix!r}.")
            continue

        # One file per unique (nn, nn_code) pair; skip rows without code
        code_df = df[["nn", "nn_code"]].dropna(subset=["nn_code"]).drop_duplicates("nn")
        if code_df.empty:
            print(f"  No source code stored for prefix {prefix!r}.")
            continue

        for _, row in code_df.iterrows():
            model_name = row["nn"]
            code = row["nn_code"]
            out_file = dest / f"{model_name}.py"
            out_file.write_text(str(code), encoding="utf-8")
            print(f"  Saved: {out_file}")
            found_any = True

    if not found_any:
        print("\nNo model code was extracted.")
    else:
        print(f"\nAll code files saved to: {dest.resolve()}")


def cmd_best_accuracy(task: str | None, csv_out: str | None, txt_out: str | None) -> None:
    """
    Show the best accuracy for each model, grouped by dataset. Optionally
    filter by task and export to CSV or TXT.
    """
    df = _load_df(task=task, only_best=False)

    required = {"nn", "dataset", "accuracy"}
    if not required.issubset(df.columns):
        print(f"ERROR: Missing columns: {required - set(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if task:
        mask = df["task"].str.lower() == task.lower()
        df = df[mask]
        if df.empty:
            print(f"No data found for task: {task!r}")
            return

    summary = (
        df.groupby(["dataset", "nn"], sort=True)["accuracy"]
        .max()
        .reset_index()
        .rename(columns={"nn": "Model", "dataset": "Dataset", "accuracy": "Best Accuracy"})
        .sort_values(["Dataset", "Best Accuracy"], ascending=[True, False])
    )
    summary["Best Accuracy"] = summary["Best Accuracy"].round(6)

    header = f"Best accuracy per model grouped by dataset"
    if task:
        header += f" (task: {task})"
    print(f"\n{header}")
    print("─" * 70)
    print(summary.to_string(index=False))
    print()

    if csv_out:
        _export_table(summary, csv_out, "Table")

    if txt_out:
        with open(txt_out, "w", encoding="utf-8") as fh:
            fh.write(header + "\n")
            fh.write("─" * 70 + "\n")
            fh.write(summary.to_string(index=False))
            fh.write("\n")
        print(f"TXT saved -> {txt_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lemur_query.py",
        description=(
            "LEMUR NN Dataset — user-facing extraction tool.\n\n"
            "Run with no arguments to list all available tasks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--task", "-t",
        metavar="TASK",
        help="Filter by task name (e.g. 'image classification'). "
             "Shows models, datasets, and max accuracy for that task.",
    )
    p.add_argument(
        "--best-accuracy", "-b",
        action="store_true",
        help="Show the best accuracy per model grouped by dataset. "
             "Combine with --task to restrict to one task.",
    )
    p.add_argument(
        "--code", "-c",
        metavar="PREFIX[,PREFIX,...]",
        help="Comma-separated model name prefix(es). "
             "Retrieves and saves the source code of all matching models.",
    )
    p.add_argument(
        "--out-dir", "-o",
        metavar="DIR",
        default=".",
        help="Directory to save extracted model code files (default: current directory).",
    )
    p.add_argument(
        "--csv",
        metavar="FILE",
        help="Export the output table to a CSV file.",
    )
    p.add_argument(
        "--txt",
        metavar="FILE",
        help="Export the output table to a plain-text file.",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # --code: extract model source code
    if args.code:
        prefixes = [p.strip() for p in args.code.split(",") if p.strip()]
        cmd_extract_code(prefixes, args.out_dir if args.out_dir != "." else None)
        return 0

    # --best-accuracy: best accuracy per model grouped by dataset
    if args.best_accuracy:
        cmd_best_accuracy(args.task, args.csv, args.txt)
        return 0

    # --task: summary for a specific task
    if args.task:
        cmd_task_summary(args.task, args.csv, args.txt)
        return 0

    # No arguments: list all tasks
    cmd_list_tasks()
    return 0


if __name__ == "__main__":
    sys.exit(main())
