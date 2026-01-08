import json

from ab.nn.util.Const import *
from ab.nn.util.Util import is_full_config, str_not_none
from ab.nn.util.db.Init import sql_conn, close_conn
from ab.nn.util.db.Write import init_population

from ab.nn.util.db.Query import *

init_population()


def query_cursor_cols_rows(*q) -> tuple[list, list]:
    conn, cursor = sql_conn()
    cursor.execute(*q)
    rows = cursor.fetchall()
    # Extract column names from cursor description.
    columns = [col[0] for col in cursor.description]
    close_conn(conn)
    return columns, rows


def query_rows(*q):
    conn, cursor = sql_conn()
    cursor.execute(*q)
    rows = cursor.fetchall()
    close_conn(conn)
    return [tuple([s for s in r]) for r in rows]


def query_cols_rows(q) -> tuple[list, list]:
    rows = query_rows(q)
    # Since each row is a tuple (with one element), you can simply use row[0]
    columns = [row[0] for row in rows]
    return columns, rows


def data(only_best_accuracy: bool = False,
         task: Optional[str] = None,
         dataset: Optional[str] = None,
         metric: Optional[str] = None,
         nn: Optional[str | tuple[str]] = None,
         epoch: Optional[int] = None,
         max_rows: Optional[int] = None,
         nn_prefixes: Optional[tuple] = None,
         sql: Optional[JoinConf] = None,
         unique_nn: bool = False,
         ) -> tuple[
    dict[str, int | float | str | dict[str, int | float | str]], ...
]:
    """
    Get the NN model code and all related statistics from the database.
    
    - If only_best_accuracy == True, then for every unique combination of 
      (task, dataset, metric, nn, epoch) only the row with the highest accuracy is returned.
    - If only_best_accuracy == False, all matching rows are returned.
    - Additionally, if any of the parameters (task, dataset, metric, nn, epoch) is not None,
      the results are filtered accordingly in the SQL query.
      
    Each returned dictionary has the following keys:
      - 'task': str
      - 'dataset': str
      - 'metric': str
      - 'metric_code': str    (source code from the metric table)
      - 'nn': str or tuple[str]
      - 'nn_code': str        (source code from the nn table)
      - 'epoch': int
      - 'accuracy': float
      - 'duration': int
      - 'prm': dict           (hyperparameters, reconstructed from the "prm" table)
      - 'transform_code': str (source code from the transform table)
    """

    # Build filtering conditions based on provided parameters.
    params, where_clause = sql_where([task, dataset, metric, nn, epoch])
    if nn_prefixes:
        where_clause += ' AND (' + ' OR '.join([f"nn LIKE '{prefix}%'" for prefix in nn_prefixes]) + ')'

    source = f'(SELECT s.* FROM stat s {where_clause})'
    if unique_nn:
        source = f'(SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY nn ORDER BY accuracy DESC) rn FROM {source}) WHERE rn = 1)'
    if only_best_accuracy:
        source = """
            (WITH filtered_stat AS {source}
            SELECT f.* FROM filtered_stat f
            JOIN (
                SELECT task, dataset, metric, nn, epoch, MAX(accuracy) AS max_accuracy
                FROM filtered_stat
                GROUP BY task, dataset, metric, nn, epoch
            ) b
            ON f.task = b.task AND f.dataset = b.dataset AND f.metric = b.metric
               AND f.nn = b.nn AND f.epoch = b.epoch AND f.accuracy = b.max_accuracy
        )""".format(source=source)

    base_query = """
        SELECT s.id, s.task, s.dataset, s.metric, m.code AS metric_code, m.id AS metric_id,
               s.nn, n.code AS nn_code, n.id AS nn_id, s.epoch, s.accuracy, s.duration,
               s.prm AS prm_id, t.code AS transform_code, t.id AS transform_id, s.transform
        FROM {source} s
        LEFT JOIN nn       n ON s.nn = n.name
        LEFT JOIN metric   m ON s.metric = m.name
        LEFT JOIN transform t ON s.transform = t.name       
    """.format(source=source)

    limit_clause = str_not_none('LIMIT ', max_rows)

    # Execute a *single* query for the main stat rows
    conn = None
    try:
        conn, cur = sql_conn()
        if sql: cur.execute(f'DROP TABLE IF EXISTS {tmp_data}')
        cur.execute(f'CREATE TEMP TABLE {tmp_data} AS {base_query} ORDER BY RANDOM()' if sql else
                    f'''{base_query} 
                        ORDER BY s.task, s.dataset, s.metric, s.nn, s.epoch 
                        {limit_clause}''',
                    params)
        if sql:
            results = join_nn_query(sql, limit_clause, cur)
        else:
            results = fill_hyper_prm(cur)
        return tuple(results)
    finally:
        if conn: close_conn(conn)


def run_data(
        model_name: str | None = None,
        device_type: str | None = None,
        max_rows: int | None = None,
):
    """
    Query mobile runtime analytics from the `mobile` table with optional filters.
    Returns a tuple of dicts with columns and parsed device_analytics JSON.
    """
    params = []
    filters = []
    if model_name is not None:
        filters.append('model_name = ?')
        params.append(model_name)
    if device_type is not None:
        filters.append('device_type = ?')
        params.append(device_type)

    where_clause = (' WHERE ' + ' AND '.join(filters)) if filters else ''
    limit_clause = (' LIMIT ' + str(max_rows)) if max_rows else ''

    conn, cur = sql_conn()
    try:
        cur.execute(
            f"""
            SELECT id, model_name, device_type, os_version, valid, emulator, error_message, duration,
                   iterations, unit, cpu_duration, cpu_min_duration, cpu_max_duration, cpu_std_dev, cpu_error,
                   gpu_duration, gpu_min_duration, gpu_max_duration, gpu_std_dev, gpu_error,
                   npu_duration, npu_min_duration, npu_max_duration, npu_std_dev, npu_error,
                   total_ram_kb, free_ram_kb, available_ram_kb, cached_kb,
                   in_dim_0, in_dim_1, in_dim_2, in_dim_3, device_analytics_json
            FROM {run_table}
            {where_clause}
            ORDER BY model_name
            {limit_clause}
            """,
            params,
        )
        rows = cur.fetchall()
        columns = [c[0] for c in cur.description]
        results = []
        for r in rows:
            rec = dict(zip(columns, r))
            try:
                if rec.get('device_analytics_json'):
                    rec['device_analytics'] = json.loads(rec['device_analytics_json'])
            except Exception:
                rec['device_analytics'] = None
            rec.pop('device_analytics_json', None)
            results.append(rec)
        return tuple(results)
    finally:
        close_conn(conn)


def sql_where(value_list):
    filters = []
    params = []
    for nm, v in zip(main_columns_ext, value_list):
        if v is not None:
            if isinstance(v, tuple):
                phs = ",".join("?" for _ in v)
                filters.append(f"s.{nm} in ({phs})")
                params.extend(v)
            else:
                filters.append(f"s.{nm} = ?")
                params.append(v)
    return params, ' WHERE ' + ' AND '.join(filters) if filters else ''


def remaining_trials(config_ext, n_optuna_trials) -> tuple[int, int]:
    """
    Calculate the number of remaining Optuna trials for a given model configuration by querying the database.
    
    Instead of reading trial counts from a file, we query the database to count all trial records
    for the specified model (identified by model_name). The trial_file parameter is retained for
    interface compatibility but is not used.
    
    If n_optuna_trials is negative, its absolute value is taken as the required number of additional trials.
    Otherwise, the function computes:
    
        remaining_trials = max(0, n_optuna_trials - n_passed_trials)
    
    :param config_ext: Tuple of names (Task, Dataset, Metric, Model, Epoch).
    :param n_optuna_trials: Target number of trials. If negative, its absolute value specifies the additional trials required.
    :return: A tuple (n_remaining_trials, n_passed_trials) where:
             - n_remaining_trials is the number of new trials to run (or 0 if none remain).
             - n_passed_trials is the number of trials already recorded in the database for this model.
    """

    conn, cursor = sql_conn()
    params, where_clause = sql_where(config_ext)
    cursor.execute('SELECT COUNT(*) AS trial_count FROM stat s' + where_clause, params)
    row = cursor.fetchone()
    if row:
        # Convert the tuple row to a dict
        columns = [col[0] for col in cursor.description]
        row_dict = dict(zip(columns, row))
        n_passed_trials = row_dict.get("trial_count", 0)
    else:
        n_passed_trials = 0

    if n_optuna_trials < 0:
        n_remaining_trials = abs(n_optuna_trials)
    else:
        n_remaining_trials = max(0, n_optuna_trials - n_passed_trials)

    if n_passed_trials > 0:
        print(f"Model '{config_ext[-2]}' has {n_passed_trials} recorded trial(s), {n_remaining_trials} remaining.")

    close_conn(conn)
    return n_remaining_trials, n_passed_trials


def supported_transformers() -> list[str]:
    """
    Returns a list of all transformer names available in the database.
    
    The function queries the 'transform' table for all records and extracts the 'name'
    field from each row.
    """
    return query_cols_rows("SELECT name FROM transform")[0]


def unique_configs(patterns: list[tuple[str, ...]]) -> list[list[str]]:
    """
    Returns a list of unique configuration strings from the database that match at least one of the input patterns.
    
    A configuration string is constructed by concatenating the 'task', 'dataset', 'metric', and 'nn'
    fields from the 'stat' table using the configuration splitter defined in your constants.
    
    :param patterns: A tuple of configuration prefix patterns.
    :return: A list of unique configuration strings that start with any of the provided patterns.
    """
    matched_configs = []
    for pattern in patterns:
        pattern = list(filter(None, pattern))
        params, where_clause = sql_where(pattern)
        # params = params if not params else params[:-1] + [params[-1] + '*']
        rows = query_rows(f"SELECT DISTINCT {', '.join(main_columns)} FROM stat s" + where_clause, params)
        if not rows and is_full_config(pattern):
            rows = [tuple(pattern)]
        matched_configs = matched_configs + rows
    return list(set(matched_configs))
