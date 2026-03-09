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

#-------Helpers--------

def resolve_work_table(cur: Cursor, preferred: str = tmp_data, fallback: str = "stat") -> str:
    cur.execute(
        "SELECT 1 FROM sqlite_temp_master WHERE type IN ('table','view') AND name = ?",
        (preferred,),
    )
    if cur.fetchone():
        return preferred

    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (preferred,),
    )
    if cur.fetchone():
        return preferred

    return fallback
def _is_real_table(cur: Cursor, name: str) -> bool:
    cur.execute(
        "SELECT type FROM sqlite_temp_master WHERE name = ?",
        (name,),
    )
    row = cur.fetchone()
    if row:
        return row[0] == "table"

    cur.execute(
        "SELECT type FROM sqlite_master WHERE name = ?",
        (name,),
    )
    row = cur.fetchone()
    return bool(row and row[0] == "table")

def build_stat_filters_sql(sql: JoinConf, alias: str = "b") -> tuple[str, list]:
    """
    WHERE clause for task/dataset/metric filters.
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


def _anchor_candidates_table() -> str:
    return f"{tmp_data}_anchor_candidates"


def _ensure_temp_work_indexes(cur: Cursor, work_table: str) -> None:
    if work_table != tmp_data or not _is_real_table(cur, work_table):
        return
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{work_table}_id ON {work_table}(id)")
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{work_table}_nn_acc_epoch "
        f"ON {work_table}(nn, accuracy DESC, epoch ASC, id)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{work_table}_acc_nn_epoch "
        f"ON {work_table}(accuracy DESC, nn, epoch ASC, id)"
    )


def _prepare_anchor_candidates(cur: Cursor, sql: "JoinConf", work_table: str) -> str:
    cand_table = _anchor_candidates_table()
    where_sql, params = build_stat_filters_sql(sql, alias="s")

    _ensure_temp_work_indexes(cur, work_table)
    cur.execute(f"DROP TABLE IF EXISTS {cand_table}")
    cur.execute(
        f"""
        CREATE TEMP TABLE {cand_table} AS
        WITH base AS (
          SELECT s.id, s.nn, s.accuracy, s.epoch
          FROM {work_table} s
          {where_sql}
        ),
        best_per_nn AS (
          SELECT b.id,
                 b.nn,
                 b.accuracy,
                 b.epoch,
                 ROW_NUMBER() OVER (
                   PARTITION BY b.nn
                   ORDER BY b.accuracy DESC, b.epoch ASC, b.nn ASC
                 ) AS rn
          FROM base b
        )
        SELECT p.id, p.nn, p.accuracy, p.epoch, m.hashvalues AS hv
        FROM best_per_nn p
        JOIN nn_minhash m ON m.nn = p.nn
        WHERE p.rn = 1
        """,
        params,
    )
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{cand_table}_nn ON {cand_table}(nn)")
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{cand_table}_acc_epoch "
        f"ON {cand_table}(accuracy DESC, epoch ASC, nn)"
    )
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{cand_table}_id ON {cand_table}(id)")
    return cand_table
#---------DB MinHash + SQLite UDF----------
def _anchor_band_db(
    *,
    cur: Cursor,
    sql: "JoinConf",
    work_table: str,
    anchor_nn: str,
    min_j: float,
    max_j: float,
    limit_k: int,
) -> None:
    """
    Requires:
      - table nn_minhash(nn, hashvalues, ...)
      - SQLite UDF: jaccard_blobs(blob_a, blob_b)
    """

    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nn_minhash'"
    )
    if not cur.fetchone():
        raise RuntimeError("nn_minhash table missing in this db_file. You're on the wrong DB.")

    # Ensure anchor exists
    cur.execute("SELECT 1 FROM nn_minhash WHERE nn = ? LIMIT 1", (anchor_nn,))
    if not cur.fetchone():
        raise ValueError(f"anchor_nn='{anchor_nn}' missing in nn_minhash")

    cand_table = _anchor_candidates_table()
    anchor_hv_row = cur.execute(
        f"SELECT hv FROM {cand_table} WHERE nn = ? LIMIT 1",
        (anchor_nn,),
    ).fetchone()
    anchor_hv = anchor_hv_row[0] if anchor_hv_row else cur.execute(
        "SELECT hashvalues FROM nn_minhash WHERE nn = ?",
        (anchor_nn,),
    ).fetchone()[0]

    cur.execute(
        f"""
        WITH
        scored AS (
          SELECT
            c.id,
            c.nn,
            c.accuracy,
            c.epoch,
            jaccard_blobs(?, c.hv) AS j
          FROM {cand_table} c
          WHERE c.nn <> ?
        )
        SELECT d.*, s.j AS anchor_jaccard
        FROM scored s
        JOIN {work_table} d ON d.id = s.id
        WHERE s.j IS NOT NULL
          AND s.j >= ? AND s.j < ?
        ORDER BY s.accuracy DESC, s.j DESC, s.nn ASC, s.epoch ASC
        LIMIT ?
        """,
        [anchor_hv, anchor_nn, float(min_j), float(max_j), int(limit_k)],
    )
#---------Band Aware Anchor Selection---------
def _resolve_anchor(
    cur: Cursor,
    sql: "JoinConf",
    work_table: str,
    *,
    min_j: float,
    max_j: float,
    limit_k: int,
    max_trials: int = 50,
) -> str:
    # If anchor_nn is explicitly given
    if sql.anchor_nn:
        return str(sql.anchor_nn)

    # Ensure nn_minhash exists
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='nn_minhash'")
    if not cur.fetchone():
        raise RuntimeError("nn_minhash table missing in this db_file")

    cand_table = _anchor_candidates_table()

    # Candidate anchors: top unique NNs from the prepared candidate table.
    anchors = cur.execute(
        f"""
        SELECT c.nn
        FROM {cand_table} c
        ORDER BY c.accuracy DESC, c.epoch ASC, c.nn ASC
        LIMIT ?
        """,
        [int(max_trials)],
    ).fetchall()

    if not anchors:
        raise ValueError(
            "Could not auto-select anchor_nn — no nns with stored MinHash "
            f"found for task={sql.task}, dataset={sql.dataset}, metric={sql.metric}"
        )

    attempted_bands: list[tuple[str, float, float]] = []
    if sql.similarity_band in SIM_BANDS:
        ordered_band_names = [
            band_name
            for band_name, _ in sorted(
                SIM_BANDS.items(),
                key=lambda item: (float(item[1][0]), float(item[1][1])),
                reverse=True,
            )
        ]
        start_idx = ordered_band_names.index(str(sql.similarity_band))
        for band_name in ordered_band_names[start_idx:]:
            band_min, band_max = SIM_BANDS[band_name]
            attempted_bands.append((band_name, float(band_min), float(band_max)))
    else:
        attempted_bands.append(("custom", float(min_j), float(max_j)))

    # For each candidate anchor, check the requested band first, then lower bands if needed.
    for band_name, band_min, band_max in attempted_bands:
        for (a_nn,) in anchors:
            anchor_hv_row = cur.execute(
                f"SELECT hv FROM {cand_table} WHERE nn = ? LIMIT 1",
                (a_nn,),
            ).fetchone()
            if not anchor_hv_row:
                continue
            anchor_hv = anchor_hv_row[0]
            cnt = cur.execute(
                f"""
                WITH
                scored AS (
                  SELECT jaccard_blobs(?, c.hv) AS j
                  FROM {cand_table} c
                  WHERE c.nn <> ?
                )
                SELECT COUNT(*)
                FROM scored
                WHERE j IS NOT NULL AND j >= ? AND j < ?
                """,
                [anchor_hv, a_nn, band_min, band_max],
            ).fetchone()[0]

            if cnt >= int(limit_k):
                return a_nn

    attempted_desc = ", ".join(
        f"{band_name}=[{band_min}, {band_max})"
        for band_name, band_min, band_max in attempted_bands
    )

    raise ValueError(
        f"Auto-anchor failed: no anchor among top {len(anchors)} models has >= {limit_k} "
        f"neighbors in attempted bands {attempted_desc} for task={sql.task}, "
        f"dataset={sql.dataset}, metric={sql.metric}."
    )
#-----JoinConf-----

@dataclass(frozen=True)
class JoinConf:

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
    similarity_mode: str = "none"  # "none" | "anchor_band_db_minhash"

    # Curriculum knobs
    anchor_nn: Optional[str] = None                 # required if anchor_band
    similarity_band: Optional[str] = None           # "high"|"medium"|"low"|"very_low"|None
    min_arch_jaccard: float = 0.0
    max_arch_jaccard: float = 1.0

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

        if self.similarity_mode not in ("none", "anchor_band_db_minhash"):
            raise ValueError("similarity_mode must be 'none' or 'anchor_band_db_minhash'.")

        if self.similarity_mode == "anchor_band_db_minhash":
            # anchor_nn is now optional — resolved automatically if absent
            if self.anchor_nn is not None and not isinstance(self.anchor_nn, str):
                raise ValueError("anchor_nn must be a string if provided.")
            mn, mx = band_to_range(self.similarity_band, self.min_arch_jaccard, self.max_arch_jaccard)
            if not (0.0 <= mn <= mx <= 1.0000001):
                raise ValueError(f"Invalid arch band: min={mn}, max={mx}")


        if self.similarity_band is not None and self.similarity_band not in SIM_BANDS:
            raise ValueError(f"Invalid similarity_band: {self.similarity_band}")



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

def join_nn_query(sql: JoinConf,limit_clause: Optional[str], cur):
    if sql.similarity_mode == "anchor_band_db_minhash":
        return join_nn_query_anchor_otf(sql,limit_clause, cur)

    # similarity_mode == "none"
    if sql.num_joint_nns > 1 and not sql.same_columns and not sql.diff_columns:
        # NEW: pure SQL variable-N selection
        return join_nn_query_sql_Var_num(sql,limit_clause, cur)

    # fallback: legacy pairwise logic
    return join_nn_query_legacy(sql,limit_clause, cur)

def join_nn_query_anchor_otf(sql: JoinConf, limit_clause: Optional[str], cur: Cursor) -> list[dict]:
    sql.validate()
    n = int(sql.num_joint_nns)
    work = resolve_work_table(cur, preferred=tmp_data, fallback="stat")

    min_j, max_j = band_to_range(sql.similarity_band, sql.min_arch_jaccard, sql.max_arch_jaccard)
    _prepare_anchor_candidates(cur, sql, work)

    # Band-aware auto-anchor selection
    anchor = _resolve_anchor(cur, sql, work, min_j=min_j, max_j=max_j, limit_k=n)

    _anchor_band_db(
        cur=cur,
        sql=sql,
        work_table=work,
        anchor_nn=anchor,
        min_j=min_j,
        max_j=max_j,
        limit_k=n,
    )

    selected = fill_hyper_prm(cur, num_joint_nns=1)
    attach_arch_summaries(selected)

    if selected:
        dm = selected[0].setdefault("diversity_meta", {})
        if isinstance(dm, dict):
            dm["curriculum_meta"] = {
                "mode": "anchor_band_db_minhash",
                "anchor_nn": anchor,
                "auto_anchor": sql.anchor_nn is None,
                "band": sql.similarity_band,
                "min_j": min_j,
                "max_j": max_j,
                "work_table": work,
            }
    return selected


"""SQL-only variable-N model selection.
No similarity constraints. One row per model"""

def join_nn_query_sql_Var_num(sql: JoinConf,limit_clause:Optional[str], cur: Cursor) -> list[dict]:
    sql.validate()
    n = int(sql.num_joint_nns)

    work = resolve_work_table(cur, preferred=tmp_data, fallback="stat")

    where_sql, params = build_stat_filters_sql(sql, alias="b")

    cur.execute(
        f"""
        WITH base AS (
          SELECT *
          FROM {work} b
          {where_sql}
        ),
        best_per_nn AS (
          SELECT b.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY b.nn
                   ORDER BY b.accuracy DESC, b.epoch ASC, b.nn ASC
                 ) AS rn
          FROM base b
        )
        SELECT *
        FROM best_per_nn
        WHERE rn = 1
        ORDER BY accuracy DESC
        LIMIT ?
        """,
        params + [n],
    )

    return fill_hyper_prm(cur, num_joint_nns=1)

def join_nn_query_legacy(sql: JoinConf,limit_clause:Optional[str], cur):
    if _is_real_table(cur, tmp_data):
        cur.execute(f'CREATE INDEX IF NOT EXISTS i_id ON {tmp_data}(id)')

    if _is_real_table(cur, tmp_data):
        cols = set()
        if sql.same_columns:
            cols.update(sql.same_columns)
        if sql.diff_columns:
            cols.update(sql.diff_columns)
        if sql.enhance_nn:
            cols.add("accuracy")

        if cols:
            t = ", ".join(sorted(cols))
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_tmp_join ON {tmp_data}({t})"
            )

    q_list = []
    for c in (sql.same_columns or ()):
        q_list.append(f'd2.{c} = d1.{c}')

    for c in (sql.diff_columns or ()):
        q_list.append(f'd2.{c} != d1.{c}')

    if sql.enhance_nn:
        q_list.append(f'd2.accuracy > d1.accuracy')
    where_clause = 'WHERE ' + ' AND '.join(q_list) if q_list else ''

    cur.execute(f'''
WITH matches AS (
  SELECT
      d1.*,
      (
          SELECT d2.id
          FROM {tmp_data} d2
          {where_clause}
          LIMIT {sql.num_joint_nns - 1}
      ) AS matched_id
  FROM {tmp_data} d1
)
SELECT
    m.*,
    d2.nn       AS nn_2,
    d2.nn_code  AS nn_code_2,
    d2.metric AS metric_2,
    d2.metric_code  AS metric_code_2,
    d2.transform_code  AS transform_code_2,
    d2.prm_id AS prm_id_2,
    d2.accuracy AS accuracy_2,
    d2.duration AS duration_2,    
    d2.epoch AS epoch_2    
FROM matches m
LEFT JOIN {tmp_data} d2 ON d2.id = m.matched_id{limit_clause}''')
    return fill_hyper_prm(cur, sql.num_joint_nns)

#------Hyperparameter assembly------
def fill_hyper_prm(cur: Cursor, num_joint_nns=1, include_nn_stats=False) -> list[dict]:
    rows = cur.fetchall()
    if not rows: return []  # short-circuit for an empty result
    columns = [c[0] for c in cur.description]

    # Bulk-load *all* hyperparameters for the retrieved stat_ids
    from collections import defaultdict
    prm_by_uid: dict[str, dict[str, int | float | str]] = defaultdict(dict)
    prm_ids: list[str] = []
    seen_prm_ids: set[str] = set()
    for r in rows:
        rec = dict(zip(columns, r))
        for i in range(1, num_joint_nns + 1):
            key = 'prm_id' if i == 1 else f'prm_id_{i}'
            prm_id = rec.get(key)
            if prm_id and prm_id not in seen_prm_ids:
                seen_prm_ids.add(prm_id)
                prm_ids.append(prm_id)

    chunk_size = 500
    for start in range(0, len(prm_ids), chunk_size):
        chunk = prm_ids[start:start + chunk_size]
        placeholders = ', '.join(['?'] * len(chunk))
        cur.execute(f"SELECT uid, name, value FROM prm WHERE uid IN ({placeholders})", chunk)
        for uid, name, value in cur.fetchall():
            prm_by_uid[uid][name] = value

    # Assemble the final result
    results: list[dict] = []
    for r in rows:
        rec = dict(zip(columns, r))
        rec['prm'] = prm_by_uid.get(rec['prm_id'], {})
        for i in range(2, num_joint_nns + 1):
            i = str(i)
            rec['prm_' + i] = prm_by_uid.get(rec['prm_id_' + i], {})
        rec.pop('transform', None)

        # Parse nn_stats_meta JSON if present
        if include_nn_stats and 'nn_stats_meta' in rec and rec['nn_stats_meta']:
            try:
                import json
                rec['nn_stats_meta'] = json.loads(rec['nn_stats_meta'])
            except Exception:
                rec['nn_stats_meta'] = None

        results.append(rec)
    return results
