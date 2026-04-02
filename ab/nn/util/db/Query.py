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

    where_sql, where_params = build_stat_filters_sql(sql, alias="d")

    cur.execute(
        f"""
        WITH base AS (
          SELECT d.*
          FROM {work_table} d
          {where_sql}
        ),
        best_per_nn AS (
          SELECT b.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY b.nn
                   ORDER BY b.accuracy DESC, b.epoch ASC, b.nn ASC
                 ) AS rn
          FROM base b
        ),
        anchor AS (
          SELECT hashvalues AS a_hv
          FROM nn_minhash
          WHERE nn = ?
        ),
        cand AS (
          SELECT p.id, p.nn, p.accuracy, p.epoch, m.hashvalues AS hv
          FROM best_per_nn p
          JOIN nn_minhash m ON m.nn = p.nn
          WHERE p.rn = 1 AND p.nn <> ?
        ),
        scored AS (
          SELECT
            c.id,
            c.nn,
            c.accuracy,
            c.epoch,
            jaccard_blobs((SELECT a_hv FROM anchor), c.hv) AS j
          FROM cand c
        )
        SELECT d.*, s.j AS anchor_jaccard
        FROM scored s
        JOIN {work_table} d ON d.id = s.id
        WHERE s.j IS NOT NULL
          AND s.j >= ? AND s.j < ?
        ORDER BY s.accuracy DESC, s.j DESC, s.nn ASC, s.epoch ASC
        LIMIT ?
        """,
        [*where_params, anchor_nn, anchor_nn, float(min_j), float(max_j), int(limit_k)],
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

    where_sql, params = build_stat_filters_sql(sql, alias="s")

    # Candidate anchors: top accuracy models that also have minhash
    anchors = cur.execute(
        f"""
        SELECT s.nn
        FROM {work_table} s
        JOIN nn_minhash m ON m.nn = s.nn
        {where_sql}
        ORDER BY s.accuracy DESC
        LIMIT ?
        """,
        params + [int(max_trials)],
    ).fetchall()

    if not anchors:
        raise ValueError(
            "Could not auto-select anchor_nn — no nns with stored MinHash "
            f"found for task={sql.task}, dataset={sql.dataset}, metric={sql.metric}"
        )

    # For each candidate anchor, check if the requested band has enough neighbors.
    for (a_nn,) in anchors:
        cnt = cur.execute(
            f"""
            WITH base AS (
              SELECT s.*
              FROM {work_table} s
              {where_sql}
            ),
            best_per_nn AS (
              SELECT b.*,
                     ROW_NUMBER() OVER (
                       PARTITION BY b.nn
                       ORDER BY b.accuracy DESC, b.epoch ASC, b.nn ASC
                     ) AS rn
              FROM base b
            ),
            anchor AS (
              SELECT hashvalues AS a_hv
              FROM nn_minhash
              WHERE nn = ?
            ),
            cand AS (
              SELECT p.nn, m.hashvalues AS hv
              FROM best_per_nn p
              JOIN nn_minhash m ON m.nn = p.nn
              WHERE p.rn = 1 AND p.nn <> ?
            ),
            scored AS (
              SELECT jaccard_blobs((SELECT a_hv FROM anchor), c.hv) AS j
              FROM cand c
            )
            SELECT COUNT(*)
            FROM scored
            WHERE j IS NOT NULL AND j >= ? AND j < ?
            """,
            [*params, a_nn, a_nn, float(min_j), float(max_j)],
        ).fetchone()[0]

        if cnt >= int(limit_k):
            return a_nn

    # Raise error if no anchor had enough neighbors in that band.
    raise ValueError(
        f"Auto-anchor failed: no anchor among top {len(anchors)} models has >= {limit_k} "
        f"neighbors in band [{min_j}, {max_j}) for task={sql.task}, dataset={sql.dataset}, metric={sql.metric}."
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


def _fetch_code_lookup(cur: Cursor, table: str, names: set) -> dict:
    if not names:
        return {}
    lookup = {}
    ordered_names = sorted(names)
    chunk_size = 900
    for start in range(0, len(ordered_names), chunk_size):
        chunk = ordered_names[start:start + chunk_size]
        placeholders = ', '.join(['?'] * len(chunk))
        cur.execute(f"SELECT name, code, id FROM {table} WHERE name IN ({placeholders})", chunk)
        for name, code, uid in cur.fetchall():
            lookup[name] = (code, uid)
    return lookup


def _attach_legacy_code_fields(cur: Cursor, results: list) -> None:
    if not results:
        return
    nn_map = _fetch_code_lookup(
        cur, "nn",
        {r["nn"] for r in results if r.get("nn")} | {r["nn_2"] for r in results if r.get("nn_2")},
    )
    metric_map = _fetch_code_lookup(
        cur, "metric",
        {r["metric"] for r in results if r.get("metric")} | {r["metric_2"] for r in results if r.get("metric_2")},
    )
    transform_map = _fetch_code_lookup(
        cur, "transform",
        {r.get("transform_name") for r in results if r.get("transform_name")} |
        {r.get("transform_name_2") for r in results if r.get("transform_name_2")},
    )
    for rec in results:
        if rec.get("nn") in nn_map:
            rec["nn_code"], rec["nn_id"] = nn_map[rec["nn"]]
        if rec.get("nn_2") in nn_map:
            rec["nn_code_2"], _ = nn_map[rec["nn_2"]]
        if rec.get("metric") in metric_map:
            rec["metric_code"], rec["metric_id"] = metric_map[rec["metric"]]
        if rec.get("metric_2") in metric_map:
            rec["metric_code_2"], _ = metric_map[rec["metric_2"]]
        if rec.get("transform_name") in transform_map:
            rec["transform_code"], rec["transform_id"] = transform_map[rec["transform_name"]]
        if rec.get("transform_name_2") in transform_map:
            rec["transform_code_2"], _ = transform_map[rec["transform_name_2"]]
        rec.pop("transform_name", None)
        rec.pop("transform_name_2", None)


def join_nn_query_legacy(sql: JoinConf, limit_clause: Optional[str], cur, include_nn_stats: bool = False):
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

        # Pure diff-only path (no equality constraints, no accuracy threshold):
        # the correlated subquery ORDER BY is (accuracy DESC, epoch ASC, ...).
        # Without a covering index this is O(N²); with it SQLite stops after
        # the first 1-2 rows that satisfy the inequality condition → O(N).
        # Skip when same_columns exist: idx_tmp_join already covers those queries.
        if not sql.same_columns and not sql.enhance_nn:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_tmp_acc_order "
                f"ON {tmp_data}(accuracy DESC, epoch ASC, id ASC)"
            )

    q_list = []
    for c in (sql.same_columns or ()):
        q_list.append(f'd2.{c} = d1.{c}')

    for c in (sql.diff_columns or ()):
        q_list.append(f'd2.{c} != d1.{c}')

    if sql.enhance_nn:
        q_list.append(f'd2.accuracy > d1.accuracy')
    where_clause = 'WHERE ' + ' AND '.join(q_list) if q_list else ''

    nn_stats_select = ""
    nn_stats_join = ""
    if include_nn_stats:
        nn_stats_select = """
    ns.total_params AS nn_total_params, ns.trainable_params AS nn_trainable_params,
    ns.frozen_params AS nn_frozen_params, ns.total_layers AS nn_total_layers,
    ns.leaf_layers AS nn_leaf_layers, ns.max_depth AS nn_max_depth,
    ns.flops AS nn_flops, ns.model_size_mb AS nn_model_size_mb,
    ns.buffer_size_mb AS nn_buffer_size_mb, ns.total_memory_mb AS nn_total_memory_mb,
    ns.dropout_count AS nn_dropout_count, ns.has_attention AS nn_has_attention,
    ns.has_residual_connections AS nn_has_residual, ns.is_resnet_like AS nn_is_resnet_like,
    ns.is_vgg_like AS nn_is_vgg_like, ns.is_inception_like AS nn_is_inception_like,
    ns.is_densenet_like AS nn_is_densenet_like, ns.is_unet_like AS nn_is_unet_like,
    ns.is_transformer_like AS nn_is_transformer_like, ns.is_mobilenet_like AS nn_is_mobilenet_like,
    ns.is_efficientnet_like AS nn_is_efficientnet_like, ns.code_length AS nn_code_length,
    ns.num_classes_defined AS nn_num_classes, ns.num_functions_defined AS nn_num_functions,
    ns.uses_sequential AS nn_uses_sequential, ns.uses_modulelist AS nn_uses_modulelist,
    ns.uses_moduledict AS nn_uses_moduledict, ns.meta_json AS nn_stats_meta,
    ns.error AS nn_stats_error,
"""
        nn_stats_join = f"LEFT JOIN nn_stat ns ON m.nn = ns.nn_name AND m.prm_id = ns.prm_id"

    cur.execute(f'''
WITH matches AS (
  SELECT
      d1.id, d1.task, d1.dataset, d1.metric,
      NULL AS metric_code, NULL AS metric_id,
      d1.nn, NULL AS nn_code, NULL AS nn_id,
      d1.epoch, d1.accuracy, d1.duration,
      d1.prm AS prm_id,
      NULL AS transform_code, NULL AS transform_id,
      d1.transform AS transform_name,
      (
          SELECT d2.id
          FROM {tmp_data} d2
          {where_clause}
          ORDER BY d2.accuracy DESC, d2.epoch ASC, d2.nn ASC, d2.id ASC
          LIMIT {sql.num_joint_nns - 1}
      ) AS matched_id
  FROM {tmp_data} d1
  {limit_clause}
)
SELECT
    m.id, m.task, m.dataset, m.metric,
    m.metric_code, m.metric_id,
    m.nn, m.nn_code, m.nn_id,
    m.epoch, m.accuracy, m.duration, m.prm_id,
    m.transform_code, m.transform_id, m.transform_name,
    {nn_stats_select}
    d2.nn       AS nn_2,
    NULL        AS nn_code_2,
    d2.metric   AS metric_2,
    NULL        AS metric_code_2,
    NULL        AS transform_code_2,
    d2.prm      AS prm_id_2,
    d2.accuracy AS accuracy_2,
    d2.duration AS duration_2,
    d2.epoch    AS epoch_2,
    d2.transform AS transform_name_2
FROM matches m
LEFT JOIN {tmp_data} d2 ON d2.id = m.matched_id
{nn_stats_join}
{limit_clause}''')
    results = fill_hyper_prm(cur, sql.num_joint_nns, include_nn_stats=include_nn_stats)
    _attach_legacy_code_fields(cur, results)
    return results

#------Hyperparameter assembly------
def fill_hyper_prm(cur: Cursor, num_joint_nns=1, include_nn_stats=False) -> list[dict]:
    rows = cur.fetchall()
    if not rows: return []  # short-circuit for an empty result
    columns = [c[0] for c in cur.description]

    # Bulk-load only the hyperparameters needed for the retrieved result rows.
    from collections import defaultdict
    prm_by_uid: dict[str, dict[str, int | float | str]] = defaultdict(dict)

    col_idx = {name: idx for idx, name in enumerate(columns)}
    needed_prm_ids: set = set()
    for col in ["prm_id", *[f"prm_id_{i}" for i in range(2, num_joint_nns + 1)]]:
        idx = col_idx.get(col)
        if idx is not None:
            for row in rows:
                uid = row[idx]
                if uid is not None:
                    needed_prm_ids.add(uid)
    if needed_prm_ids:
        chunk_size = 900
        for i in range(0, len(needed_prm_ids), chunk_size):
            chunk = sorted(needed_prm_ids)[i:i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
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

        # Parse nn_stats_meta JSON if present
        if include_nn_stats and 'nn_stats_meta' in rec and rec['nn_stats_meta']:
            try:
                import json
                rec['nn_stats_meta'] = json.loads(rec['nn_stats_meta'])
            except Exception:
                rec['nn_stats_meta'] = None

        results.append(rec)
    return results