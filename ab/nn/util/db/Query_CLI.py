from __future__ import annotations

import argparse
import json

from ab.nn.api import data
from ab.nn.util.db.Query import JoinConf


def _csv_to_tuple(v: str | None):
    if not v:
        return None
    parts = [p.strip() for p in v.split(",") if p.strip()]
    return tuple(parts) if parts else None


def _compact_view(rows: list[dict], *, keep_pairs_top: int = 5) -> list[dict]:
    """
    Make output human-readable:
    - drop huge code fields
    - show key stats + diversity explanation
    """
    drop_keys = {"nn_code", "metric_code", "transform_code"}
    out = []

    for r in rows:
        rr = {k: v for k, v in r.items() if k not in drop_keys}

        # Keep only the important bits from diversity_meta
        dm = rr.get("diversity_meta")
        if isinstance(dm, dict):
            pw = dm.get("pairwise_summary")
            if isinstance(pw, dict):
                pw = dict(pw)
                # shrink pairs list
                if isinstance(pw.get("pairs_top"), list):
                    pw["pairs_top"] = pw["pairs_top"][:keep_pairs_top]
                dm = dict(dm)
                dm["pairwise_summary"] = pw

            rr["diversity_meta"] = {
                "use_arch_diversity": dm.get("use_arch_diversity"),
                "max_arch_jaccard": dm.get("max_arch_jaccard"),
                "min_arch_jaccard": dm.get("min_arch_jaccard"),
                "requested_k": dm.get("requested_k"),
                "selected_k": dm.get("selected_k"),
                "skipped_missing_signature": dm.get("skipped_missing_signature"),
                "skipped_too_similar": dm.get("skipped_too_similar"),
                "fallback_filled": dm.get("fallback_filled"),
                "pairwise_summary": dm.get("pairwise_summary"),
            }

        # Optional: shrink prm too (can be huge)
        prm = rr.get("prm")
        if isinstance(prm, dict) and len(prm) > 30:
            rr["prm"] = {k: prm[k] for k in list(prm.keys())[:30]}
            rr["prm_truncated"] = True

        out.append(rr)

    return out


def main():
    ap = argparse.ArgumentParser(
        prog="Query_CLI",
        description="Phase-1: Retrieve N models for same task/dataset/metric with optional architecture diversity (code_minhash).",
    )

    # Slice filters (functional equivalence)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--metric", type=str, default=None)
    ap.add_argument("--nn", type=str, default=None)
    ap.add_argument("--epoch", type=int, default=None)

    # Output / sizing
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--summary", action="store_true", help="Compact output: hide large code fields, show diversity summary.")
    ap.add_argument("--include-nn-stats", action="store_true", help="Join nn_stat columns from DB (enables arch_summary).")

    # JoinConf core
    ap.add_argument("--num-joint-nns", type=int, required=True)

    ap.add_argument("--same-columns", type=str, default=None)
    ap.add_argument("--diff-columns", type=str, default=None)
    ap.add_argument("--enhance-nn", action="store_true")

    # Architecture diversity (Option 1)
    ap.add_argument("--use-arch-diversity", action="store_true")
    ap.add_argument("--max-arch-jaccard", type=float, default=0.85)
    ap.add_argument("--min-arch-jaccard", type=float, default=0.0)
    ap.add_argument("--arch-stat-dir", type=str, default="ab/nn/stat/nn")
    ap.add_argument("--overfetch-factor", type=int, default=20)
    ap.add_argument("--no-fallback-fill", action="store_true")

    args = ap.parse_args()

    sql = JoinConf(
        num_joint_nns=args.num_joint_nns,
        same_columns=_csv_to_tuple(args.same_columns),
        diff_columns=_csv_to_tuple(args.diff_columns),
        enhance_nn=True if args.enhance_nn else None,
        use_arch_diversity=bool(args.use_arch_diversity),
        min_arch_jaccard=float(args.min_arch_jaccard),
        max_arch_jaccard=float(args.max_arch_jaccard),
        arch_stat_dir=str(args.arch_stat_dir),
        overfetch_factor=int(args.overfetch_factor),
        allow_fallback_fill=not bool(args.no_fallback_fill),
    )

    df = data(
        task=args.task,
        dataset=args.dataset,
        metric=args.metric,
        nn=args.nn,
        epoch=args.epoch,
        max_rows=None,
        sql=sql,
        unique_nn=False,
        only_best_accuracy=False,
        include_nn_stats=bool(args.include_nn_stats),
    )

    rows = df.to_dict(orient="records")[: int(args.limit)]

    print(f"\n[RESULT] Retrieved {len(rows)} rows\n")

    if args.summary:
        rows = _compact_view(rows)

    if args.pretty or args.summary:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(rows)


if __name__ == "__main__":
    main()
