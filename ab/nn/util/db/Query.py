from __future__ import annotations

import json
from dataclasses import dataclass
from sqlite3 import Cursor
from typing import Optional

from ab.nn.util.Const import main_columns_ext, tmp_data
import re

#-----curriculum bands-----

SIM_BANDS: dict[str, tuple[float, float]] = {
    "high": (0.95, 1.0000001),
    "medium": (0.85, 0.95),
    "low": (0.60, 0.85),
    "very_low": (0.0, 0.60),
}

TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[^\s]")

def _tokenize(code: str) -> list[str]:
    return TOKEN_RE.findall(code or "")

def _shingles(tokens: list[str], n: int = 7) -> list[str]:
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _minhash_hashvalues(code: str, *, num_perm: int = 128, shingle_n: int = 7) -> list[int]:
    """
    Return MinHash.hashvalues as Python integers (length=num_perm)
    Uses datasketch.MinHash
    """
    try:
        from datasketch import MinHash
    except Exception as e:
        raise RuntimeError("datasketch is required for similarity_mode='anchor_band_*'") from e

    mh = MinHash(num_perm=num_perm)
    for sh in _shingles(_tokenize(code), n=shingle_n):
        mh.update(sh.encode("utf-8", errors="ignore"))
    return [int(x) for x in mh.hashvalues.tolist()]

def _minhash_jaccard(hv_a: list[int], hv_b: list[int]) -> float:
    if not hv_a or not hv_b or len(hv_a) != len(hv_b):
        return 0.0
    eq = 0
    for x, y in zip(hv_a, hv_b):
        if x == y:
            eq += 1
    return eq / len(hv_a)

def band_to_range(band: Optional[str], default_min: float, default_max: float) -> tuple[float, float]:
    if band is None:
        return float(default_min), float(default_max)
    if band not in SIM_BANDS:
        raise ValueError(f"Invalid similarity_band: {band}")
    mn, mx = SIM_BANDS[band]
    return float(mn), float(mx)

#-----Helpers-----

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

def _anchor_band_otf(
    *,
    cur: Cursor,
    sql: "JoinConf",
    work_table: str,
    anchor_nn: str,
    min_j: float,
    max_j: float,
    limit_k: int,
    num_perm: int = 128,
    shingle_n: int = 7,
) -> None:
    # Get best row per nn within filters, from tmp_data/stat
    where_sql, where_params = build_stat_filters_sql(sql, alias="b")
    cur.execute(
        f"""
        WITH base AS (
          SELECT * FROM {work_table} b {where_sql}
        ),
        best_per_nn AS (
          SELECT b.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY b.nn
                   ORDER BY b.accuracy DESC, b.epoch ASC, b.nn ASC
                 ) AS rn
          FROM base b
        )
        SELECT * FROM best_per_nn WHERE rn = 1
        """,
        where_params,
    )
    rows = cur.fetchall()
    if not rows:
        cur.execute("SELECT 1 WHERE 0")
        return

    cols = [c[0] for c in cur.description]
    recs = [dict(zip(cols, r)) for r in rows]

    # Find anchor
    anchor = next((r for r in recs if r.get("nn") == anchor_nn), None)
    if anchor is None:
        raise ValueError(f"anchor_nn='{anchor_nn}' not present in candidate set for given filters")

    anchor_code = anchor.get("nn_code")
    if not isinstance(anchor_code, str) or not anchor_code:
        raise RuntimeError("nn_code missing in tmp_data rows; cannot compute similarity")

    hv_cache: dict[str, list[int]] = {}
    def hv_for(r: dict) -> list[int]:
        nn_name = str(r.get("nn"))
        hv = hv_cache.get(nn_name)
        if hv is None:
            code = r.get("nn_code") or ""
            hv = _minhash_hashvalues(code, num_perm=num_perm, shingle_n=shingle_n) if code else []
            hv_cache[nn_name] = hv
        return hv

    anchor_hv = hv_for(anchor)

    # Score candidates
    scored: list[tuple[float, float, str, int, str]] = []
    # (accuracy, jaccard, nn, epoch, id)
    for r in recs:
        if r.get("nn") == anchor_nn:
            continue
        hv = hv_for(r)
        if not hv:
            continue
        j = _minhash_jaccard(anchor_hv, hv)
        if j >= min_j and j < max_j:
            scored.append((
                float(r.get("accuracy") or 0.0),
                float(j),
                str(r.get("nn") or ""),
                int(r.get("epoch") or 0),
                str(r.get("id") or ""),
            ))

    if not scored:
        cur.execute("SELECT 1 WHERE 0")
        return

    scored.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    scored = scored[: int(limit_k)]

    # Materialize ranking temp table
    cur.execute("DROP TABLE IF EXISTS tmp_anchor_band")
    cur.execute(
        """
        CREATE TEMP TABLE tmp_anchor_band (
          id TEXT PRIMARY KEY,
          rank INTEGER NOT NULL,
          anchor_jaccard REAL NOT NULL
        )
        """
    )
    cur.executemany(
        "INSERT INTO tmp_anchor_band(id, rank, anchor_jaccard) VALUES (?, ?, ?)",
        [(rid, i, j) for i, (_, j, _, _, rid) in enumerate(scored, start=1)],
    )

    # Final select
    cur.execute(
        f"""
        SELECT d.*, t.anchor_jaccard
        FROM {work_table} d
        JOIN tmp_anchor_band t ON t.id = d.id
        ORDER BY t.rank ASC
        """
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
    similarity_mode: str = "none"  # "none" | "anchor_band" | "anchor_band_otf"

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

        if self.similarity_mode not in ("none", "anchor_band_otf"):
            raise ValueError("similarity_mode must be 'none' or 'anchor_band_otf'.")

        if self.similarity_mode == "anchor_band_otf":
            if not self.anchor_nn or not isinstance(self.anchor_nn, str):
                raise ValueError("anchor_nn is required for similarity_mode='anchor_band_otf'.")
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

def join_nn_query(sql: JoinConf, cur):
    if sql.similarity_mode == "anchor_band_otf":
        return join_nn_query_anchor_otf(sql, cur)

    # similarity_mode == "none"
    if sql.num_joint_nns > 1 and not sql.same_columns and not sql.diff_columns:
        # NEW: pure SQL variable-N selection
        return join_nn_query_sql_Var_num(sql, cur)

    # fallback: legacy pairwise logic
    return join_nn_query_legacy(sql, cur)


def join_nn_query_anchor_otf(sql: JoinConf, cur: Cursor) -> list[dict]:
    sql.validate()
    n = int(sql.num_joint_nns)

    # Use tmp_data if present,else fallback to stat.
    work = resolve_work_table(cur, preferred=tmp_data, fallback="stat")
    if work != tmp_data:
        raise RuntimeError(
            f"Expected '{tmp_data}' to exist, but it doesn't. "
            "NNGPT requires nn_code/metric_code/transform_code columns which stat doesn't provide. "
            "Rebuild tmp_data or run the pipeline that generates it."
        )

    anchor = str(sql.anchor_nn)
    min_j, max_j = band_to_range(sql.similarity_band, sql.min_arch_jaccard, sql.max_arch_jaccard)

    _anchor_band_otf(cur=cur, sql=sql, work_table=work, anchor_nn=anchor, min_j=min_j, max_j=max_j, limit_k=n)

    selected = fill_hyper_prm(cur, num_joint_nns=1)
    attach_arch_summaries(selected)

    if selected:
        dm = selected[0].setdefault("diversity_meta", {})
        if isinstance(dm, dict):
            dm["curriculum_meta"] = {
                "mode": "anchor_band_on_the_fly",
                "anchor_nn": anchor,
                "band": sql.similarity_band,
                "min_j": min_j,
                "max_j": max_j,
                "work_table": work,
            }
    return selected

"""SQL-only variable-N model selection.
No similarity constraints. One row per model"""

def join_nn_query_sql_Var_num(sql: JoinConf, cur: Cursor) -> list[dict]:
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

def join_nn_query_legacy(sql: JoinConf, cur):
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
LEFT JOIN {tmp_data} d2 ON d2.id = m.matched_id''')
    return fill_hyper_prm(cur, sql.num_joint_nns)

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