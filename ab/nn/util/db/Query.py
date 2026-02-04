from dataclasses import dataclass, field
from sqlite3 import Cursor
from typing import Optional

from ab.nn.util.Const import main_columns_ext, tmp_data

<<<<<<< Updated upstream
=======
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
>>>>>>> Stashed changes

@dataclass(frozen=True)
class JoinConf:
    # Obligatory parameter
    num_joint_nns: int  # Required

    # Optional parameters
    same_columns: Optional[tuple[str, ...]] = None
    diff_columns: Optional[tuple[str, ...]] = None
    enhance_nn: Optional[bool] = None  # Parameter for accuracy improvement check

    supported_columns = main_columns_ext

    def validate(self):
        # Validate the 'num_joint_nns' (though it's obligatory, it's good to check its range if needed)
        if self.num_joint_nns < 1:
            raise ValueError("The number of joint NNs ('num_joint_nns') must be a positive integer greater than 1.")

        # If 'diff_columns' or 'same_columns' are provided, ensure they contain valid column names
        if self.diff_columns:
            for column_name in self.diff_columns:
                if column_name not in self.supported_columns:
                    raise ValueError(f"Unsupported column name in 'diff_columns': {column_name}")

        if self.same_columns:
            for column_name in self.same_columns:
                if column_name not in self.supported_columns:
                    raise ValueError(f"Unsupported column name in 'same_columns': {column_name}")

        # Validation for the new 'first_model_lower_accuracy' parameter
        if self.enhance_nn is not None:
            if not isinstance(self.enhance_nn, bool):
                raise ValueError("'first_model_lower_accuracy' must be a boolean value (True or False).")

        # You can add more validation for the newly added parameter if needed


def join_nn_query(sql: JoinConf, limit_clause: Optional[str], cur):
    cur.execute(f'CREATE INDEX IF NOT EXISTS i_id ON {tmp_data}(id)')
    if sql.same_columns or sql.diff_columns or sql.enhance_nn:
        t = tuple({*(sql.same_columns or set()), *(sql.diff_columns or set())}) + ('accuracy',) if sql.enhance_nn else ()
        cur.execute(f'CREATE INDEX idx_tmp_join ON {tmp_data}{t}')

    q_list = []
    for c in sql.same_columns:
        q_list.append(f'd2.{c} = d1.{c}')
    for c in sql.diff_columns:
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
LEFT JOIN {tmp_data} d2 ON d2.id = m.matched_id {limit_clause}''')
    return fill_hyper_prm(cur, sql.num_joint_nns)


def fill_hyper_prm(cur: Cursor, num_joint_nns=1, include_nn_stats=False) -> list[dict]:
    rows = cur.fetchall()
    if not rows: return []  # short-circuit for an empty result
    columns = [c[0] for c in cur.description]

    # Bulk-load *all* hyperparameters for the retrieved stat_ids
    from collections import defaultdict
    prm_by_uid: dict[str, dict[str, int | float | str]] = defaultdict(dict)

    cur.execute(f"SELECT uid, name, value FROM prm")
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
