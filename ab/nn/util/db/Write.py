import sys

from tqdm import tqdm

from ab.nn.util.Util import *
from ab.nn.util.db.Init import init_db, sql_conn, close_conn
from ab.nn.util.hf.DB_from_HF import db_from_hf
from ab.nn.util.Const import nn_stat_table, stat_run_tflite_fp32_dir, stat_run_tflite_int8_dir, run_table, tflite_table
from ab.nn.util.db.build_nn_similarity import upsert_minhash, upsert_minhash_batch


def init_population():
    if not db_file.exists():
        if stat_dir.exists():
            init_db()
            json_train_to_db()
            try:
                json_run_tflite_to_db()
            except Exception as e:
                print(f"TFLite runtime analytics import failed: {e}")
            try:
                json_nn_to_db()
            except Exception as e:
                print(f"NN statistics import failed: {e}")
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
    sub_configs = [d.name for d in stat_train_dir.iterdir() if d.is_dir()]

    print(f"Importing all statistics from JSON files in {stat_train_dir} into database {db_file} ...")

    for sub_config_str in tqdm(sub_configs):
        model_stat_dir = stat_train_dir / sub_config_str

        for epoch_file in Path(model_stat_dir).iterdir():
            model_stat_file = model_stat_dir / epoch_file
            epoch = int(epoch_file.stem)

            try:
                with open(model_stat_file, 'r', encoding='utf-8', errors='replace') as f:
                    for trial in json.load(f):
                        _, _, metric, nn = sub_config = conf_to_names(sub_config_str)
                        populate_code_table('nn', cursor, name=nn)
                        # Handle comma-separated metrics (e.g., "bleu,meteor,cider")
                        for single_metric in metric.split(','):
                            populate_code_table('metric', cursor, name=single_metric.strip())
                        populate_code_table('transform', cursor, name=trial['transform'])
                        save_stat(sub_config + (epoch,), trial, cursor)
            except Exception as e:
                print(f"Warning: skipping JSON file {model_stat_file}: {e}", file=sys.stderr)
    close_conn(conn)
    print("All statistics reloaded successfully.")


def json_nn_to_db():
    """
    Import NN statistics from JSON files in stat/nn into the `nn_stat` table.
    Each JSON file contains model statistics with the nn_name derived from filename
    and prm_id stored within the JSON.
    """
    if not stat_nn_dir.exists():
        return

    json_files = list(stat_nn_dir.glob("*.json"))
    if not json_files:
        return

    print(f"Importing NN statistics from {len(json_files)} JSON files in {stat_nn_dir} into database...")
    conn, cursor = sql_conn()

    success_count = 0
    error_count = 0

    minhash_batch = []

    for json_file in tqdm(json_files, desc="Importing NN stats"):
        try:
            # Extract nn_name from filename (without .json extension)
            nn_name = json_file.stem

            # Read JSON file
            with open(json_file, 'r', encoding='utf-8', errors='replace') as f:
                stats = json.load(f)

            # Get prm_id from the JSON content
            prm_id = stats.get('prm_id')

            if not prm_id:
                error_count += 1
                continue

            # Save to database using existing function
            if save_nn_stat(nn_name, prm_id, stats):
                success_count += 1
            else:
                error_count += 1
            # Collect Minhash for batch write
            cm = stats.get("code_minhash")
            if isinstance(cm, dict) and cm.get("available") and isinstance(cm.get("hashvalues"), list):
                minhash_batch.append({
                    "nn": nn_name,
                    "hashvalues": cm["hashvalues"],
                    "num_perm": cm.get("num_perm", 128),
                    "shingle_n": cm.get("shingle_n", 7),
                })

        except Exception as e:
            error_count += 1
            print(f"Warning: skipping NN statistics JSON file {json_file}: {e}", file=sys.stderr)
    # Single batch write — one transaction for all vectors
    if minhash_batch:
        inserted = upsert_minhash_batch(conn, minhash_batch)
        print(f"MinHash vectors upserted: {inserted}")

    close_conn(conn)

    print(f"NN statistics import complete: {success_count} succeeded, {error_count} failed.")


def save_results(config_ext: tuple[str, str, str, str, int], prm: dict):
    """
    Save Optuna study results for a given model to SQLite DB
    :param config_ext: Tuple of names (Task, Dataset, Metric, Model, Epoch).
    :param prm: Dictionary of all saved parameters.
    """
    conn, cursor = sql_conn()
    _, _, metric, nn = config_ext[:4]
    populate_code_table('nn', cursor, name=nn)
    for single_metric in metric.split(','):
        populate_code_table('metric', cursor, name=single_metric.strip())
    save_stat(config_ext, prm, cursor)
    close_conn(conn)


def save_nn(nn_code: str, task: str, dataset: str, metric: str, epoch: int, prm: dict, force_name = None):
    conn, cursor = sql_conn()
    nn = code_to_db(cursor, 'nn', code=nn_code, force_name=force_name)
    save_stat((task, dataset, metric, nn, epoch), prm, cursor)

    close_conn(conn)
    return nn


def save_nn_stat(nn_name: str, prm_id: str, stats: dict):
    """
    Save NN statistics to the database.

    :param nn_name: Name of the neural network model
    :param prm_id: Parameter ID (hyperparameter configuration identifier)
    :param stats: Dictionary containing all statistics (should match the structure from analyze_model_comprehensive)
    """
    conn, cursor = sql_conn()

    try:
        # Generate unique ID for this statistics entry
        id_val = uuid4([nn_name, prm_id])

        # Check if entry already exists
        cursor.execute(f"SELECT id FROM {nn_stat_table} WHERE nn_name = ? AND prm_id = ?", (nn_name, prm_id))
        existing_entry = cursor.fetchone()

        if existing_entry:
            # Update existing entry
            if 'error' in stats:
                # If there's an error, only update the error field
                cursor.execute(
                    f"UPDATE {nn_stat_table} SET error = ? WHERE nn_name = ? AND prm_id = ?",
                    (stats.get('error'), nn_name, prm_id)
                )
            else:
                # Extract meta information as JSON
                meta_json = json.dumps(stats.get('meta', {}))

                # Update all fields
                cursor.execute(f"""
                UPDATE {nn_stat_table} SET
                    total_layers = ?,
                    leaf_layers = ?,
                    max_depth = ?,
                    total_params = ?,
                    trainable_params = ?,
                    frozen_params = ?,
                    flops = ?,
                    model_size_mb = ?,
                    buffer_size_mb = ?,
                    total_memory_mb = ?,
                    dropout_count = ?,
                    has_attention = ?,
                    has_residual_connections = ?,
                    is_resnet_like = ?,
                    is_vgg_like = ?,
                    is_inception_like = ?,
                    is_densenet_like = ?,
                    is_unet_like = ?,
                    is_transformer_like = ?,
                    is_mobilenet_like = ?,
                    is_efficientnet_like = ?,
                    code_length = ?,
                    num_classes_defined = ?,
                    num_functions_defined = ?,
                    uses_sequential = ?,
                    uses_modulelist = ?,
                    uses_moduledict = ?,
                    meta_json = ?,
                    error = NULL
                WHERE nn_name = ? AND prm_id = ?
                """, (
                    stats.get('total_layers'),
                    stats.get('leaf_layers'),
                    stats.get('max_depth'),
                    stats.get('total_params'),
                    stats.get('trainable_params'),
                    stats.get('frozen_params'),
                    stats.get('flops'),
                    stats.get('model_size_mb'),
                    stats.get('buffer_size_mb'),
                    stats.get('total_memory_mb'),
                    stats.get('dropout_count'),
                    stats.get('has_attention'),
                    stats.get('has_residual_connections'),
                    stats.get('is_resnet_like'),
                    stats.get('is_vgg_like'),
                    stats.get('is_inception_like'),
                    stats.get('is_densenet_like'),
                    stats.get('is_unet_like'),
                    stats.get('is_transformer_like'),
                    stats.get('is_mobilenet_like'),
                    stats.get('is_efficientnet_like'),
                    stats.get('code_length'),
                    stats.get('num_classes_defined'),
                    stats.get('num_functions_defined'),
                    stats.get('uses_sequential'),
                    stats.get('uses_modulelist'),
                    stats.get('uses_moduledict'),
                    meta_json,
                    nn_name,
                    prm_id
                ))
        else:
            # Insert new entry
            if 'error' in stats:
                # If there's an error, insert minimal entry with error
                cursor.execute(
                    f"""
                    INSERT INTO {nn_stat_table} (id, nn_name, prm_id, error)
                    VALUES (?, ?, ?, ?)
                    """,
                    (id_val, nn_name, prm_id, stats.get('error'))
                )
            else:
                # Extract meta information as JSON
                meta_json = json.dumps(stats.get('meta', {}))

                # Insert all fields
                cursor.execute(f"""
                INSERT INTO {nn_stat_table} (
                    id, nn_name, prm_id,
                    total_layers, leaf_layers, max_depth,
                    total_params, trainable_params, frozen_params,
                    flops, model_size_mb, buffer_size_mb, total_memory_mb,
                    dropout_count, has_attention, has_residual_connections,
                    is_resnet_like, is_vgg_like, is_inception_like,
                    is_densenet_like, is_unet_like, is_transformer_like,
                    is_mobilenet_like, is_efficientnet_like,
                    code_length, num_classes_defined, num_functions_defined,
                    uses_sequential, uses_modulelist, uses_moduledict,
                    meta_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    id_val, nn_name, prm_id,
                    stats.get('total_layers'),
                    stats.get('leaf_layers'),
                    stats.get('max_depth'),
                    stats.get('total_params'),
                    stats.get('trainable_params'),
                    stats.get('frozen_params'),
                    stats.get('flops'),
                    stats.get('model_size_mb'),
                    stats.get('buffer_size_mb'),
                    stats.get('total_memory_mb'),
                    stats.get('dropout_count'),
                    stats.get('has_attention'),
                    stats.get('has_residual_connections'),
                    stats.get('is_resnet_like'),
                    stats.get('is_vgg_like'),
                    stats.get('is_inception_like'),
                    stats.get('is_densenet_like'),
                    stats.get('is_unet_like'),
                    stats.get('is_transformer_like'),
                    stats.get('is_mobilenet_like'),
                    stats.get('is_efficientnet_like'),
                    stats.get('code_length'),
                    stats.get('num_classes_defined'),
                    stats.get('num_functions_defined'),
                    stats.get('uses_sequential'),
                    stats.get('uses_modulelist'),
                    stats.get('uses_moduledict'),
                    meta_json
                ))

        close_conn(conn)
        return True
    except Exception as e:
        close_conn(conn)
        print(f"Error saving NN statistics to database: {e}")
        return False


def json_run_tflite_to_db():
    """
    Import runtime analytics from tflite JSON files in stat/run/tflite/fp32 and stat/run/tflite/int8.
    Adds precision_type column to indicate whether the data is from fp32 or int8.
    """
    conn, cursor = sql_conn()
    
    tflite_dirs = [
        (stat_run_tflite_fp32_dir, 'fp32'),
        (stat_run_tflite_int8_dir, 'int8')
    ]
    
    total_files = 0
    # load mapping from all_models.json for each precision
    accuracy_maps: dict[str, dict] = {}
    for tflite_dir, precision_type in tflite_dirs:
        if not tflite_dir.exists():
            continue

        # attempt to read accuracy/transform mapping
        map_path = tflite_dir / 'all_models.json'
        if map_path.exists():
            try:
                with open(map_path, 'r', encoding='utf-8') as mf:
                    accuracy_maps[precision_type] = json.load(mf)
            except Exception as e:
                print(f"Warning: failed to load mapping from {map_path}: {e}", file=sys.stderr)
                accuracy_maps[precision_type] = {}
        else:
            accuracy_maps[precision_type] = {}

        # Count total JSON files for progress bar
        json_files = list(tflite_dir.rglob('*.json'))
        total_files += len(json_files)
    
    if total_files == 0:
        close_conn(conn)
        return
    
    print(f"Importing runtime analytics from {total_files} tflite JSON files...")
    
    processed = 0
    with tqdm(total=total_files, desc="Importing tflite runtime data") as pbar:
        for tflite_dir, precision_type in tflite_dirs:
            if not tflite_dir.exists():
                continue
            
            # Recursively find all JSON files
            for json_file in tflite_dir.rglob('*.json'):
                # Skip all_models.json - it's a mapping, not a runtime record
                if json_file.name == 'all_models.json':
                    pbar.update(1)
                    processed += 1
                    continue
                
                try:
                    with open(json_file, 'r', encoding='utf-8', errors='replace') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Warning: skipping tflite JSON file {json_file}: {e}", file=sys.stderr)
                    pbar.update(1)
                    processed += 1
                    continue
                
                # Extract data
                model_name = data.get('model_name')
                # lookup accuracy/transform
                acc_info = accuracy_maps.get(precision_type, {}).get(model_name, {})
                accuracy = acc_info.get('accuracy')
                transform_code = acc_info.get('transform')
                device_type = data.get('device_type')
                os_version = data.get('os_version')
                valid = bool(data.get('valid')) if 'valid' in data else None
                emulator = bool(data.get('emulator')) if 'emulator' in data else None
                error_message = data.get('error_message')
                duration = data.get('duration')
                device_analytics = data.get('device_analytics')
                analytics_json = json.dumps(device_analytics) if device_analytics is not None else None
                
                # Extract extra fields
                extra_vals = {f: data.get(f) for f in [
                    'iterations', 'unit',
                    'cpu_duration', 'cpu_min_duration', 'cpu_max_duration', 'cpu_std_dev', 'cpu_error',
                    'gpu_duration', 'gpu_min_duration', 'gpu_max_duration', 'gpu_std_dev', 'gpu_error',
                    'npu_duration', 'npu_min_duration', 'npu_max_duration', 'npu_std_dev', 'npu_error',
                    'total_ram_kb', 'free_ram_kb', 'available_ram_kb', 'cached_kb',
                    'in_dim_0', 'in_dim_1', 'in_dim_2', 'in_dim_3'
                ]}
                
                # Generate unique ID
                id_val = uuid4([json_file.name, model_name, device_type, os_version, duration, precision_type])
                
                # Prepare columns and values (no accuracy/transform in run table)
                columns = [
                    'id', 'model_name', 'device_type', 'os_version', 'valid', 'emulator', 'error_message', 'duration',
                    'iterations', 'unit', 'cpu_duration', 'cpu_min_duration', 'cpu_max_duration', 'cpu_std_dev', 'cpu_error',
                    'gpu_duration', 'gpu_min_duration', 'gpu_max_duration', 'gpu_std_dev', 'gpu_error',
                    'npu_duration', 'npu_min_duration', 'npu_max_duration', 'npu_std_dev', 'npu_error',
                    'total_ram_kb', 'free_ram_kb', 'available_ram_kb', 'cached_kb',
                    'in_dim_0', 'in_dim_1', 'in_dim_2', 'in_dim_3',
                    'device_analytics_json', 'precision_type'
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
                    analytics_json, precision_type
                ]
                
                try:
                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO {run_table}
                        ({col_names})
                        VALUES ({placeholders})
                        """,
                        values,
                    )
                except Exception as e:
                    print(f"Warning: failed to insert tflite data from {json_file}: {e}", file=sys.stderr)
                
                # Insert into tflite table if accuracy/transform data exists
                if accuracy is not None or transform_code is not None:
                    try:
                        tflite_id = uuid4([json_file.name, model_name, accuracy, transform_code, precision_type])
                        cursor.execute(
                            f"""
                            INSERT OR REPLACE INTO {tflite_table}
                            (id, model_name, accuracy, transform, precision_type)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (tflite_id, model_name, accuracy, transform_code, precision_type),
                        )
                    except Exception as e:
                        print(f"Warning: failed to insert tflite metadata for {model_name}: {e}", file=sys.stderr)
                
                pbar.update(1)
                processed += 1
    
    close_conn(conn)


init_population()
