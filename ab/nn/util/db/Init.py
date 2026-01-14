import sqlite3
from os import makedirs
from pathlib import Path

from ab.nn.util.Const import param_tables, db_file, db_dir, main_tables, code_tables, dependent_tables, all_tables, index_colum, run_table, nn_stat_table


def sql_conn():
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row  # Enable row access
    return conn, conn.cursor()


def close_conn(conn):
    conn.commit()
    conn.close()


def create_code_table(name, cursor):
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {name} (
        name TEXT PRIMARY KEY,
        code TEXT NOT NULL,
        id TEXT NOT NULL)""")

def create_param_table(name, cursor):
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {name} (
            uid   TEXT NOT NULL,
            name  TEXT NOT NULL,
            value OBJECT NOT NULL
        )
    """)



def init_db():
    """
    Initialize the SQLite database, create tables, and add indexes for optimized reads.
    """
    makedirs(Path(db_dir).absolute(), exist_ok=True)
    conn, cursor = sql_conn()

    # Create all tables with code
    for nm in code_tables:
        create_code_table(nm, cursor)

    # Create all tables with parameters
    for nm in param_tables:
        create_param_table(nm, cursor)
        # NEW: index for fast uid look-ups
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{nm}_uid ON {nm}(uid, name, value)")


    # Create main stat tables
    for nm in main_tables:
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {nm} (
            id TEXT PRIMARY KEY,
            accuracy REAL,
            epoch INTEGER,
            duration INTEGER,
            {', '.join(index_colum)},         
        """
                       + ',\n'.join([f"FOREIGN KEY ({nm}) REFERENCES {nm} (name) ON DELETE CASCADE" for nm in code_tables] +
                                    [f"FOREIGN KEY ({nm}) REFERENCES {nm} (uid) ON DELETE CASCADE" for nm in param_tables])
                       + ')')

    # Add indexes for optimized reads
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accuracy_desc ON stat (accuracy DESC)")
    for nm in index_colum:
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{nm} ON stat ({nm})")

    # Create mobile analytics table (runtime stats)
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {run_table} (
        id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        device_type TEXT,
        os_version TEXT,
        valid BOOLEAN,
        emulator BOOLEAN,
        error_message TEXT,
        duration INTEGER,
        
        iterations INTEGER,
        unit TEXT,
        
        cpu_duration INTEGER,
        cpu_min_duration INTEGER,
        cpu_max_duration INTEGER,
        cpu_std_dev REAL,
        cpu_error TEXT,
        
        gpu_duration INTEGER,
        gpu_min_duration INTEGER,
        gpu_max_duration INTEGER,
        gpu_std_dev REAL,
        gpu_error TEXT,
        
        npu_duration INTEGER,
        npu_min_duration INTEGER,
        npu_max_duration INTEGER,
        npu_std_dev REAL,
        npu_error TEXT,
        
        total_ram_kb INTEGER,
        free_ram_kb INTEGER,
        available_ram_kb INTEGER,
        cached_kb INTEGER,
        
        in_dim_0 INTEGER,
        in_dim_1 INTEGER,
        in_dim_2 INTEGER,
        in_dim_3 INTEGER,
        
        device_analytics_json TEXT,
        FOREIGN KEY (model_name) REFERENCES nn (name) ON DELETE CASCADE
    )
    """)
    # Indexes for mobile analytics
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{run_table}_model ON {run_table} (model_name);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{run_table}_device ON {run_table} (device_type);")

    # Create NN statistics table
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {nn_stat_table} (
        id TEXT PRIMARY KEY,
        nn_name TEXT NOT NULL,
        prm_id TEXT NOT NULL,
        total_layers INTEGER,
        leaf_layers INTEGER,
        max_depth INTEGER,
        total_params INTEGER,
        trainable_params INTEGER,
        frozen_params INTEGER,
        flops INTEGER,
        model_size_mb REAL,
        buffer_size_mb REAL,
        total_memory_mb REAL,
        dropout_count INTEGER,
        has_attention INTEGER,
        has_residual_connections INTEGER,
        is_resnet_like INTEGER,
        is_vgg_like INTEGER,
        is_inception_like INTEGER,
        is_densenet_like INTEGER,
        is_unet_like INTEGER,
        is_transformer_like INTEGER,
        is_mobilenet_like INTEGER,
        is_efficientnet_like INTEGER,
        code_length INTEGER,
        num_classes_defined INTEGER,
        num_functions_defined INTEGER,
        uses_sequential INTEGER,
        uses_modulelist INTEGER,
        uses_moduledict INTEGER,
        meta_json TEXT,
        error TEXT,
        FOREIGN KEY (nn_name) REFERENCES nn (name) ON DELETE CASCADE
    )
    """)
    # Indexes for NN statistics
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{nn_stat_table}_nn ON {nn_stat_table} (nn_name);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{nn_stat_table}_prm ON {nn_stat_table} (prm_id);")

    close_conn(conn)
    print(f"Database initialized at {db_file}")


def reset_db():
    """
    Clear the database and reload all NN models and statistics.
    """
    makedirs(Path(db_dir).absolute(), exist_ok=True)
    print(f"Clearing and reloading database at {db_file}")
    conn, cursor = sql_conn()

    # Drop existing tables
    for nm in all_tables:
        cursor.execute(f"DROP TABLE IF EXISTS {nm}")
    close_conn(conn)
    init_db()
