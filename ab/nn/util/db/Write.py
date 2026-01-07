import sys

from tqdm import tqdm

from ab.nn.util.Util import *
from ab.nn.util.db.Init import init_db, sql_conn, close_conn
from ab.nn.util.hf.DB_from_HF import db_from_hf

def init_population():
    if not db_file.exists():
        if stat_dir.exists():
            init_db()
            json_train_to_db()
            try:
                json_run_to_db()
            except Exception as e:
                print(f"Runtime analytics import failed: {e}")
        else:
            db_from_hf()


def code_to_db(cursor, table_name, code=None, code_file=None, force_name = None):
    # If the model does not exist, insert it with a new UUID
    if code_file:
        nm = code_file.stem 
    elif force_name is None:
        nm = uuid4(code)
    else:
        nm = force_name
    if not code:
        with open(code_file, 'r') as file:
            code = file.read()
    id_val = uuid4(code)
    # Check if the model exists in the database
    cursor.execute(f"SELECT code FROM {table_name} WHERE name = ?", (nm,))
    existing_entry = cursor.fetchone()
    if existing_entry:
        # If model exists, update the code if it has changed
        existing_code = existing_entry[0]
        if existing_code != code:
            print(f"Updating code for model: {nm}")
            cursor.execute("UPDATE nn SET code = ?, id = ? WHERE name = ?", (code, id_val, nm))
    else:
        cursor.execute(f"INSERT INTO {table_name} (name, code, id) VALUES (?, ?, ?)", (nm, code, id_val))
    return nm


def populate_code_table(table_name, cursor, name=None):
    """
    Populate the code table with models from the appropriate directory.
    """
    code_dir = nn_path(table_name)
    code_files = [code_dir / f"{name}.py"] if name else [Path(f) for f in code_dir.iterdir() if f.is_file() and f.suffix == '.py' and f.name != '__init__.py']
    for code_file in code_files:
        code_to_db(cursor, table_name, code_file=code_file)
    # print(f"{table_name} added/updated in the `{table_name}` table: {[f.stem for f in code_files]}")


def populate_prm_table(table_name, cursor, prm, uid):
    """
    Insert every hyperparameter in its native Python type.
    The target table layout is (uid TEXT, name TEXT, value).
    """
    for nm, value in prm.items():
        cursor.execute(
            f"INSERT OR IGNORE INTO {table_name} (uid, name, value) VALUES (?, ?, ?)",
            (uid, nm, value),
        )


def save_stat(config_ext: tuple[str, str, str, str, int], prm, cursor):
    # Insert each trial into the database with epoch
    transform = prm['transform']
    uid = prm.pop('uid')
    extra_main_column_values = [prm.pop(nm, None) for nm in extra_main_columns]
    for nm in param_tables:
        populate_prm_table(nm, cursor, prm, uid)
    all_values = [transform, uid, *config_ext, *extra_main_column_values]
    cursor.execute(f"""
    INSERT INTO stat (id, transform, prm, {', '.join(main_columns_ext + extra_main_columns)}) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (uuid4(all_values), *all_values))


def json_train_to_db():
    """
    Reload all statistics into the database for all subconfigs and epochs.
    """
    conn, cursor = sql_conn()
    stat_base_path = Path(stat_train_dir)
    sub_configs = [d.name for d in stat_base_path.iterdir() if d.is_dir()]

    print(f"Importing all statistics from JSON files in {stat_train_dir} into database {db_file} ...")

    last_error = None
    for sub_config_str in tqdm(sub_configs):
        model_stat_dir = stat_base_path / sub_config_str

        for epoch_file in Path(model_stat_dir).iterdir():
            model_stat_file = model_stat_dir / epoch_file
            epoch = int(epoch_file.stem)

            with open(model_stat_file, 'r') as f:
                try:
                    for trial in json.load(f):
                        _, _, metric, nn = sub_config = conf_to_names(sub_config_str)
                        populate_code_table('nn', cursor, name=nn)
                        populate_code_table('metric', cursor, name=metric)
                        populate_code_table('transform', cursor, name=trial['transform'])
                        save_stat(sub_config + (epoch,), trial, cursor)
                except Exception as e:
                    last_error = e
                    print(f"Error: JSON file {model_stat_file}, Exception {e}", file=sys.stderr)
    close_conn(conn)
    if last_error:
        raise last_error
    print("All statistics reloaded successfully.")


def json_run_to_db():
    """
    Import runtime analytics from JSON files in stat/run into the `mobile` table.
    The run directory layout encodes task/dataset/metric/nn in the parent folder name as
    "task_dataset_nn". We parse those names using the existing config splitter, and store
    device info and analytics payload as JSON text.
    """
    conn, cursor = sql_conn()
    run_base_path = Path(stat_run_dir)
    if not run_base_path.exists():
        close_conn(conn)
        return

    run_dirs = [d for d in run_base_path.iterdir() if d.is_dir()]
    for run_dir in tqdm(run_dirs):
        # Extract model name from directory suffix if possible: e.g., '..._<model>'
        parts = run_dir.name.split(config_splitter)
        nn_name = parts[-1] if len(parts) > 1 else None

        for json_file in run_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue

            # Ensure code tables have entries for the nn if we can resolve it
            model_name = data.get('model_name') or nn_name
            if model_name:
                try:
                    populate_code_table('nn', cursor, name=model_name)
                except Exception:
                    pass

            device_type = data.get('device_type')
            os_version = data.get('os_version')
            valid = bool(data.get('valid')) if 'valid' in data else None
            emulator = bool(data.get('emulator')) if 'emulator' in data else None
            error_message = data.get('error_message')
            duration = data.get('duration')
            device_analytics = data.get('device_analytics')
            analytics_json = json.dumps(device_analytics) if device_analytics is not None else None
            
            # New fields
            extra_vals = {f: data.get(f) for f in [
                'iterations', 'unit',
                'cpu_duration', 'cpu_min_duration', 'cpu_max_duration', 'cpu_std_dev', 'cpu_error',
                'gpu_duration', 'gpu_min_duration', 'gpu_max_duration', 'gpu_std_dev', 'gpu_error',
                'npu_duration', 'npu_min_duration', 'npu_max_duration', 'npu_std_dev', 'npu_error',
                'total_ram_kb', 'free_ram_kb', 'available_ram_kb', 'cached_kb',
                'in_dim_0', 'in_dim_1', 'in_dim_2', 'in_dim_3'
            ]}

            id_val = uuid4([run_dir.name, json_file.name, model_name, device_type, os_version, duration])
            
            columns = [
                'id', 'model_name', 'device_type', 'os_version', 'valid', 'emulator', 'error_message', 'duration',
                'iterations', 'unit', 'cpu_duration', 'cpu_min_duration', 'cpu_max_duration', 'cpu_std_dev', 'cpu_error',
                'gpu_duration', 'gpu_min_duration', 'gpu_max_duration', 'gpu_std_dev', 'gpu_error',
                'npu_duration', 'npu_min_duration', 'npu_max_duration', 'npu_std_dev', 'npu_error',
                'total_ram_kb', 'free_ram_kb', 'available_ram_kb', 'cached_kb',
                'in_dim_0', 'in_dim_1', 'in_dim_2', 'in_dim_3',
                'device_analytics_json'
            ]
            
            placeholders = ', '.join(['?'] * len(columns))
            col_names = ', '.join(columns)
            
            values = [
                id_val, model_name, device_type, os_version, valid, emulator, error_message, duration,
                extra_vals['iterations'], extra_vals['unit'], 
                extra_vals['cpu_duration'], extra_vals['cpu_min_duration'], extra_vals['cpu_max_duration'], extra_vals['cpu_std_dev'], extra_vals['cpu_error'],
                extra_vals['gpu_duration'], extra_vals['gpu_min_duration'], extra_vals['gpu_max_duration'], extra_vals['gpu_std_dev'], extra_vals['gpu_error'],
                extra_vals['npu_duration'], extra_vals['npu_min_duration'], extra_vals['npu_max_duration'], extra_vals['npu_std_dev'], extra_vals['npu_error'],
                extra_vals['total_ram_kb'], extra_vals['free_ram_kb'], extra_vals['available_ram_kb'], extra_vals['cached_kb'],
                extra_vals['in_dim_0'], extra_vals['in_dim_1'], extra_vals['in_dim_2'], extra_vals['in_dim_3'],
                analytics_json
            ]

            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {run_table}
                ({col_names})
                VALUES ({placeholders})
                """,
                values,
            )
    close_conn(conn)


def save_results(config_ext: tuple[str, str, str, str, int], prm: dict):
    """
    Save Optuna study results for a given model to SQLite DB
    :param config_ext: Tuple of names (Task, Dataset, Metric, Model, Epoch).
    :param prm: Dictionary of all saved parameters.
    """
    conn, cursor = sql_conn()
    save_stat(config_ext, prm, cursor)
    close_conn(conn)


def save_nn(nn_code: str, task: str, dataset: str, metric: str, epoch: int, prm: dict, force_name = None):
    conn, cursor = sql_conn()
    nn = code_to_db(cursor, 'nn', code=nn_code, force_name=force_name)
    save_stat((task, dataset, metric, nn, epoch), prm, cursor)
    close_conn(conn)
    return nn


init_population()
