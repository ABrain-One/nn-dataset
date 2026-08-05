#!/usr/bin/env python3
"""
Five examples of api.data_withnonnullvalue() combining stat, nn_stat, and prm.

Run from repo root (with project deps installed), e.g.:
  python3 test-nonnull.py
"""
from __future__ import annotations

from ab.nn import api


def _show(df, cols: list[str], *, head: int | None = None) -> None:
    """Print selected columns; tolerate empty results or missing optional columns."""
    use = [c for c in cols if c in df.columns]
    if df.empty or not use:
        print(f"(no rows or missing columns; shape={df.shape}, columns={list(df.columns)[:20]!r} ...)")
        return
    out = df[use] if head is None else df[use].head(head)
    print(out.to_string(index=False))


def main() -> None:
    print("=" * 72)
    print("Example 1 — stat + nn_stat only (required nn columns non-NULL)")
    print("  Join: nn + prm_id match; prm dict is full (no require_prm_nonnull).")
    print("=" * 72)
    df1 = api.data_withnonnullvalue(
        include_nn_stats=False,
        require_nn_stat_nonnull=("nn_total_layers", "nn_flops"),
        require_prm_nonnull=(),
        max_rows=5,
    )
    _show(df1, ["task", "dataset", "nn", "accuracy", "nn_total_layers", "nn_flops"])
    print()

    print("=" * 72)
    print("Example 2 — stat + prm only (required hyperparameter rows non-NULL)")
    print("  No nn_stat columns; prm dict only contains listed keys.")
    print("  NOTE: With no task/dataset/nn filter this scans almost all stat rows — can take minutes.")
    print("=" * 72)
    print("  (running…)", flush=True)
    df2 = api.data_withnonnullvalue(
        require_nn_stat_nonnull=(),
        require_prm_nonnull=("lr", "batch"),
        max_rows=5,
        prm_as_columns=True,
        # Narrow the stat subquery to finish quickly, e.g. task="YourTask", dataset="YourDataset",
    )
    show2 = ["nn", "accuracy", "lr", "batch"]
    _show(df2, [c for c in show2 if c in df2.columns])
    print()

    print("=" * 72)
    print("Example 3 — stat + nn_stat + prm (all three; minimal nn_stat columns)")
    print("=" * 72)
    df3 = api.data_withnonnullvalue(
        include_nn_stats=False,
        require_nn_stat_nonnull=("nn_total_layers",),
        require_prm_nonnull=("lr", "momentum"),
        max_rows=5,
        prm_as_columns=True,
    )
    show3 = ["nn", "accuracy", "nn_total_layers", "lr", "momentum"]
    _show(df3, [c for c in show3 if c in df3.columns], head=3)
    print()

    print("=" * 72)
    print("Example 4 — stat + full nn_stat + prm (include_nn_stats=True)")
    print("  All nn_* columns; still require nn_total_layers & nn_flops non-NULL.")
    print("=" * 72)
    df4 = api.data_withnonnullvalue(
        include_nn_stats=True,
        require_nn_stat_nonnull=("nn_total_layers", "nn_flops"),
        require_prm_nonnull=("lr",),
        max_rows=3,
        prm_as_columns=True,
    )
    cols_show = ["nn", "accuracy", "nn_total_layers", "nn_flops", "nn_model_size_mb", "lr", "transform"]
    _show(df4, [c for c in cols_show if c in df4.columns])
    print()

    print("=" * 72)
    print("Example 5 — same as 3 + stat filters (task / dataset / max_rows)")
    print("  (Adjust task/dataset if your DB uses different names.)")
    print("=" * 72)
    df5 = api.data_withnonnullvalue(
        task=None,
        dataset=None,
        include_nn_stats=False,
        require_nn_stat_nonnull=("nn_dropout_count",),
        require_prm_nonnull=("lr",),
        max_rows=5,
        prm_as_columns=True,
    )
    show5 = ["task", "dataset", "nn", "epoch", "nn_dropout_count", "lr"]
    _show(df5, [c for c in show5 if c in df5.columns], head=5)


if __name__ == "__main__":
    main()
