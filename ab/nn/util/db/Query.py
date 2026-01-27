from __future__ import annotations

import json
from dataclasses import dataclass
from sqlite3 import Cursor
from typing import Optional

from ab.nn.util.Const import main_columns_ext, tmp_data

#-----curriculum bands-----

SIM_BANDS: dict[str, tuple[float, float]] = {
    "high": (0.95, 1.0000001),
    "medium": (0.85, 0.95),
    "low": (0.60, 0.85),
    "very_low": (0.0, 0.60),
}

def band_to_range(band: Optional[str], default_min: float, default_max: float) -> tuple[float, float]:
    if band is None:
        return float(default_min), float(default_max)
    if band not in SIM_BANDS:
        raise ValueError(f"Invalid similarity_band: {band}")
    mn, mx = SIM_BANDS[band]
    return float(mn), float(mx)

#-----Helpers-----
def resolve_work_table(cur: Cursor, preferred: str = "tmp_data", fallback: str = "stat") -> str:
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (preferred,),
    )
    return preferred if cur.fetchone() else fallback

def build_stat_filters_sql(sql: JoinConf, alias: str = "b") -> tuple[str, list]:
    """
    Build WHERE clause for task/dataset/metric filters.
    """
    clauses: list[str] = []
    params: list = []

    if sql.task:
        clauses.append(f"{alias}.task = ?")
        params.append(sql.task)

    if sql.dataset:
        clauses.append(f"{alias}.dataset = ?")
        params.append(sql.dataset)

    if sql.metric:
        clauses.append(f"{alias}.metric = ?")
        params.append(sql.metric)

    if not clauses:
        return "", []

    return "WHERE " + " AND ".join(clauses), params
# Anchor based curriculum band

def _anchor_band(*, cur: Cursor, sql: JoinConf, work_table: str, anchor_nn: str, min_j: float, max_j: float, limit_k: int) -> None:
    where_sql, where_params = build_stat_filters_sql(sql, alias="b")

    cur.execute(
        f"""
        WITH base AS (
          SELECT *
          FROM {work_table} b
          {where_sql}
        ),
        best_per_nn AS (
          SELECT
            b.*,
            ROW_NUMBER() OVER (
              PARTITION BY b.nn
              ORDER BY b.accuracy DESC, b.epoch ASC, b.nn ASC
            ) AS rn
          FROM base b
        ),
        unique_nn AS (
          SELECT * FROM best_per_nn WHERE rn = 1
        )
        SELECT u.*, s.jaccard AS anchor_jaccard
        FROM nn_similarity s
        JOIN unique_nn u
          ON u.nn = s.nn_b
        WHERE s.nn_a = ?
          AND s.jaccard >= ?
          AND s.jaccard <  ?
        ORDER BY u.accuracy DESC, s.jaccard DESC, u.nn ASC, u.epoch ASC
        LIMIT ?
        """,
        [*where_params, anchor_nn, float(min_j), float(max_j), int(limit_k)],
    )


#-----JoinConf-----

@dataclass(frozen=True)
class JoinConf:
    """
    Query conf (READ-ONLY):

      - Reads from tmp_data
      - Selection modes:
          * similarity_mode="none": top-K accuracy from tmp_data
          * similarity_mode="anchor_band": anchor-based curriculum via nn_similarity (SQL-only)

    """

    # Required
    num_joint_nns: int

    # Optional compatibility knobs (kept for callers)
    same_columns: Optional[tuple[str, ...]] = None
    diff_columns: Optional[tuple[str, ...]] = None
    enhance_nn: Optional[bool] = None

    task: Optional[str] = None
    dataset: Optional[str] = None
    metric: Optional[str] = None

    # Mode selection
    similarity_mode: str = "none"  # "none" | "anchor_band"

    # Curriculum knobs
    anchor_nn: Optional[str] = None                 # required if anchor_band
    similarity_band: Optional[str] = None           # "high"|"medium"|"low"|"very_low"|None
    min_arch_jaccard: float = 0.0                   # used only if similarity_band is None
    max_arch_jaccard: float = 1.0                   # used only if similarity_band is None

    # How many candidates pull from tmp_data (only used for "none"; anchor-band uses direct join)
    overfetch_factor: int = 20

    supported_columns = main_columns_ext

    def validate(self) -> None:
        if int(self.num_joint_nns) < 1:
            raise ValueError("'num_joint_nns' must be >= 1.")

        if self.diff_columns:
            for c in self.diff_columns:
                if c not in self.supported_columns:
                    raise ValueError(f"Unsupported column name in diff_columns: {c}")

        if self.same_columns:
            for c in self.same_columns:
                if c not in self.supported_columns:
                    raise ValueError(f"Unsupported column name in same_columns: {c}")

        if self.enhance_nn is not None and not isinstance(self.enhance_nn, bool):
            raise ValueError("'enhance_nn' must be boolean.")

        if int(self.overfetch_factor) < 1:
            raise ValueError("'overfetch_factor' must be >= 1.")

        if self.similarity_mode not in ("none", "anchor_band_sql"):
            raise ValueError("similarity_mode must be 'none' or 'anchor_band_sql'.")

        if self.similarity_band is not None and self.similarity_band not in SIM_BANDS:
            raise ValueError(f"Invalid similarity_band: {self.similarity_band}")

        if self.similarity_mode == "anchor_band_sql":
            if not self.anchor_nn or not isinstance(self.anchor_nn, str):
                raise ValueError("anchor_nn is required for similarity_mode='anchor_band_sql'.")

        mn, mx = band_to_range(self.similarity_band, self.min_arch_jaccard, self.max_arch_jaccard)
        if not (0.0 <= mn <= mx <= 1.0000001):
            raise ValueError(f"Invalid arch band: min={mn}, max={mx}")

#------Hyperparameter assembly------
def fill_hyper_prm(cur: Cursor, num_joint_nns: int = 1, include_nn_stats: bool = False) -> list[dict]:
    rows = cur.fetchall()
    if not rows:
        return []

    columns = [c[0] for c in cur.description]

    from collections import defaultdict
    prm_by_uid: dict[str, dict[str, int | float | str]] = defaultdict(dict)

    cur.execute("SELECT uid, name, value FROM prm")
    for uid, name, value in cur.fetchall():
        prm_by_uid[uid][name] = value

    results: list[dict] = []
    for r in rows:
        rec = dict(zip(columns, r))
        rec["prm"] = prm_by_uid.get(rec.get("prm_id"), {})
        rec.pop("transform", None)

        if include_nn_stats and isinstance(rec.get("nn_stats_meta"), str) and rec["nn_stats_meta"]:
            try:
                rec["nn_stats_meta"] = json.loads(rec["nn_stats_meta"])
            except Exception:
                rec["nn_stats_meta"] = None

        results.append(rec)

    return results

#-----Optional: architecture summaries if present in rows----

def attach_arch_summaries(selected: list[dict]) -> None:
    for r in selected:
        if "nn_total_params" not in r:
            r["arch_summary"] = {}
            continue

        r["arch_summary"] = {
            "total_params": r.get("nn_total_params"),
            "trainable_params": r.get("nn_trainable_params"),
            "total_layers": r.get("nn_total_layers"),
            "leaf_layers": r.get("nn_leaf_layers"),
            "max_depth": r.get("nn_max_depth"),
            "flops": r.get("nn_flops"),
            "model_size_mb": r.get("nn_model_size_mb"),
            "dropout_count": r.get("nn_dropout_count"),
            "has_attention": r.get("nn_has_attention"),
            "has_residual": r.get("nn_has_residual"),
            "is_resnet_like": r.get("nn_is_resnet_like"),
            "is_transformer_like": r.get("nn_is_transformer_like"),
        }

# Main query (operates on tmp_data)

def join_nn_query(sql: JoinConf, cur: Cursor) -> list[dict]:
    sql.validate()
    n = int(sql.num_joint_nns)

    # Use tmp_data if present, otherwise fall back to stat.
    work = resolve_work_table(cur, preferred=tmp_data, fallback="stat")

    # ========================================================
    # Phase 2: anchor-band SQL
    # ========================================================
    if sql.similarity_mode == "anchor_band_sql":
        anchor = str(sql.anchor_nn)
        min_j, max_j = band_to_range(sql.similarity_band, sql.min_arch_jaccard, sql.max_arch_jaccard)

        _anchor_band(cur=cur, sql=sql, work_table=work, anchor_nn=anchor, min_j=min_j, max_j=max_j, limit_k=n)

        selected = fill_hyper_prm(cur, num_joint_nns=1)
        attach_arch_summaries(selected)

        if selected:
            dm = selected[0].setdefault("diversity_meta", {})
            if isinstance(dm, dict):
                dm["curriculum_meta"] = {
                    "mode": "anchor_band_sql",
                    "anchor_nn": anchor,
                    "band": sql.similarity_band,
                    "min_j": min_j,
                    "max_j": max_j,
                    "work_table": work,
                }
        return selected

    # ========================================================
    # Default: top accuracy from work table (optionally unique nn)
    # ========================================================
    where_sql, where_params = build_stat_filters_sql(sql, alias="b")

    cand_n = max(n, 1) * int(sql.overfetch_factor)
    require_diff_nn = bool(sql.diff_columns) and ("nn" in (sql.diff_columns or ()))

    if require_diff_nn:
        where_sql, where_params = build_stat_filters_sql(sql, alias="b")
        cur.execute(
            f"""
            WITH base AS (
              SELECT * FROM {work} b
              {where_sql}
            ),
            best_per_nn AS (
              SELECT
                b.*,
                ROW_NUMBER() OVER (
                  PARTITION BY b.nn
                  ORDER BY b.accuracy DESC, b.epoch ASC, b.nn ASC
                ) AS rn
              FROM base b
            ),
            unique_nn AS (
              SELECT * FROM best_per_nn WHERE rn = 1
            )
            SELECT * FROM unique_nn
            ORDER BY accuracy DESC, nn ASC, epoch ASC
            LIMIT ?
            """,
            [*where_params, cand_n],
        )
    else:
        where_sql, where_params = build_stat_filters_sql(sql, alias="b")
        cur.execute(
            f"""
            SELECT *
            FROM {work} b
            {where_sql}
            ORDER BY b.accuracy DESC, b.nn ASC, b.epoch ASC
            LIMIT ?
            """,
            [*where_params, cand_n],
        )

    selected = fill_hyper_prm(cur, num_joint_nns=1)[:n]
    attach_arch_summaries(selected)
    return selected
