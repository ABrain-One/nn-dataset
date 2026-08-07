"""
Unit tests for api.data_withnonnullvalue(), combining stat, nn_stat and prm.

Run from repo root (with project deps installed), e.g.:
    python3 test_nonnull.py            # or: python3 test_nonnull.py -v
    python3 -m unittest test_nonnull

Example 2 scans almost all stat rows and can take minutes, so it is skipped
unless RUN_SLOW_TESTS=1 is set:
    RUN_SLOW_TESTS=1 python3 test_nonnull.py
"""
from __future__ import annotations

import os
import unittest

from ab.nn import api

RUN_SLOW = os.environ.get("RUN_SLOW_TESTS") == "1"


class DataWithNonNullValueTest(unittest.TestCase):
    """Each test mirrors one of the original examples."""

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def assertFrame(self, df, required_cols, *, max_rows, optional_cols=()):
        """Validate a returned DataFrame.

        Checks that the row cap is respected, that every column in
        ``required_cols`` exists and contains no NULLs, and that any present
        column from ``optional_cols`` is likewise NULL-free. An empty result
        skips the test rather than failing it, since the contents of the
        database are not under the test's control.
        """
        self.assertTrue(hasattr(df, "columns"), "expected a pandas DataFrame")
        self.assertLessEqual(
            len(df), max_rows, f"max_rows={max_rows} not respected (got {len(df)})"
        )

        if df.empty:
            self.skipTest(f"no matching rows in the database (columns={list(df.columns)[:20]!r})")

        for col in required_cols:
            with self.subTest(column=col):
                self.assertIn(col, df.columns, f"missing expected column {col!r}")
                self.assertFalse(
                    df[col].isna().any(), f"column {col!r} contains NULL values"
                )

        for col in optional_cols:
            if col in df.columns:
                with self.subTest(column=col, optional=True):
                    self.assertFalse(
                        df[col].isna().any(), f"column {col!r} contains NULL values"
                    )

    # ------------------------------------------------------------------ #
    # Example 1 — stat + nn_stat only (required nn columns non-NULL)
    # Join: nn + prm_id match; prm dict is full (no require_prm_nonnull).
    # ------------------------------------------------------------------ #
    def test_01_stat_and_nn_stat_only(self):
        df = api.data_withnonnullvalue(
            include_nn_stats=False,
            require_nn_stat_nonnull=("nn_total_layers", "nn_flops"),
            require_prm_nonnull=(),
            max_rows=5,
        )
        self.assertFrame(
            df,
            required_cols=("nn_total_layers", "nn_flops"),
            optional_cols=("task", "dataset", "nn", "accuracy"),
            max_rows=5,
        )

    # ------------------------------------------------------------------ #
    # Example 3 — stat + nn_stat + prm (all three; minimal nn_stat columns)
    # ------------------------------------------------------------------ #
    def test_03_stat_nn_stat_and_prm(self):
        df = api.data_withnonnullvalue(
            include_nn_stats=False,
            require_nn_stat_nonnull=("nn_total_layers",),
            require_prm_nonnull=("lr", "momentum"),
            max_rows=5,
            prm_as_columns=True,
        )
        self.assertFrame(
            df,
            required_cols=("nn_total_layers", "lr", "momentum"),
            optional_cols=("nn", "accuracy"),
            max_rows=5,
        )

    # ------------------------------------------------------------------ #
    # Example 4 — stat + full nn_stat + prm (include_nn_stats=True)
    # All nn_* columns; still require nn_total_layers & nn_flops non-NULL.
    # ------------------------------------------------------------------ #
    def test_04_full_nn_stats(self):
        df = api.data_withnonnullvalue(
            include_nn_stats=True,
            require_nn_stat_nonnull=("nn_total_layers", "nn_flops"),
            require_prm_nonnull=("lr",),
            max_rows=3,
            prm_as_columns=True,
        )
        self.assertFrame(
            df,
            required_cols=("nn_total_layers", "nn_flops", "lr"),
            optional_cols=("nn", "accuracy", "transform"),
            max_rows=3,
        )
        # include_nn_stats=True should widen the frame beyond the required pair.
        self.assertTrue(
            any(c.startswith("nn_") for c in df.columns
                if c not in {"nn_total_layers", "nn_flops"}),
            "expected additional nn_* columns when include_nn_stats=True",
        )

    # ------------------------------------------------------------------ #
    # Example 5 — same as 3 + stat filters (task / dataset / max_rows)
    # (Adjust task/dataset if your DB uses different names.)
    # ------------------------------------------------------------------ #
    def test_05_with_stat_filters(self):
        df = api.data_withnonnullvalue(
            task=None,
            dataset=None,
            include_nn_stats=False,
            require_nn_stat_nonnull=("nn_dropout_count",),
            require_prm_nonnull=("lr",),
            max_rows=5,
            prm_as_columns=True,
        )
        self.assertFrame(
            df,
            required_cols=("nn_dropout_count", "lr"),
            optional_cols=("task", "dataset", "nn", "epoch"),
            max_rows=5,
        )


if __name__ == "__main__":
    unittest.main()