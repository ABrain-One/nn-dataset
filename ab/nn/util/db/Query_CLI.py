from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Tuple

from ab.nn.util.db.Init import sql_conn, close_conn
from ab.nn.util.db.Query import JoinConf, join_nn_query


def _csv_to_tuple(v: str | None) -> Optional[tuple[str, ...]]:
    if not v:
        return None
    parts = [p.strip() for p in v.split(",") if p.strip()]
    return tuple(parts) if parts else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="General + Curriculum query runner (Phase 1 + Phase 2)"
    )

    p.add_argument(
        "--mode",
        choices=["general", "curriculum"],
        default="curriculum",
        help="general: top-K best models; curriculum: anchor+band using nn_similarity",
    )

    # Phase-1 style filters (only effective if your pipeline uses them)
    p.add_argument("--task", help="Task name filter (if supported by your DB slice)")
    p.add_argument("--dataset", help="Dataset name filter (if supported by your DB slice)")
    p.add_argument("--metric", help="Metric name filter (if supported by your DB slice)")
    p.add_argument("--nn", help="NN name filter (if supported by your DB slice)")
    p.add_argument("--epoch", type=int, help="Epoch filter (if supported by your DB slice)")

    # Phase-2 curriculum knobs
    p.add_argument("--anchor", help="Anchor model name (required for --mode=curriculum)")
    p.add_argument(
        "--band",
        choices=["high", "medium", "low", "very_low"],
        default="medium",
        help="Curriculum similarity band (curriculum mode only)",
    )

    p.add_argument("--k", type=int, default=5, help="Number of models to retrieve")
    p.add_argument("--json", action="store_true", help="Output full rows as JSON")

    # Optional: ensure unique nn in general mode (recommended)
    p.add_argument(
        "--unique-nn",
        action="store_true",
        help="General mode: return at most one row per nn (best accuracy per nn).",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Enforce anchor requirement only for curriculum mode
    if args.mode == "curriculum":
        if not args.anchor or args.anchor.strip().lower() in {"none", "null"}:
            parser.error("--anchor is required when --mode=curriculum (use a real nn_similarity.nn_a value)")

    conn, cur = sql_conn()

    try:
        # Build "same_columns" list for compatibility with your existing query config.
        # NOTE: This only helps if your slice-builder uses it.
        same_cols = []
        if args.task:
            same_cols.append("task")
        if args.dataset:
            same_cols.append("dataset")
        if args.metric:
            same_cols.append("metric")
        if args.nn:
            same_cols.append("nn")
        if args.epoch is not None:
            same_cols.append("epoch")
        same_columns: Optional[Tuple[str, ...]] = tuple(same_cols) if same_cols else None

        # General mode vs curriculum mode
        if args.mode == "general":
            conf = JoinConf(
                num_joint_nns=int(args.k),
                similarity_mode="none",
                same_columns=same_columns,
                diff_columns=("nn",) if args.unique_nn else None,
                task=args.task,
                dataset=args.dataset,
                metric=args.metric,
            )
        else:
            conf = JoinConf(
                num_joint_nns=int(args.k),
                similarity_mode="anchor_band_sql",
                anchor_nn=args.anchor,
                similarity_band=args.band,
                same_columns=same_columns,
                task=args.task,
                dataset=args.dataset,
                metric=args.metric,
            )

        rows = join_nn_query(conf, cur)

        if not rows:
            print("No models returned.", file=sys.stderr)
            sys.exit(2)

        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for i, r in enumerate(rows, 1):
                j = r.get("anchor_jaccard")
                j_str = f"{j}" if j is not None else "-"
                print(
                    f"{i:02d}  "
                    f"nn={r.get('nn')}  "
                    f"acc={r.get('accuracy')}  "
                    f"j={j_str}"
                )

    finally:
        close_conn(conn)


if __name__ == "__main__":
    main()
