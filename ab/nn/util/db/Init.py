import sqlite3
from os import makedirs
from pathlib import Path

from ab.nn.util.Const import param_tables, db_file, db_dir, main_tables, code_tables, dependent_tables, all_tables, index_colum, mobile_table


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
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{nm}_uid ON {nm}(uid, name, value);")


    # Create main stat tables
    for nm in main_tables:
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {nm} (
            id TEXT PRIMARY KEY,
            accuracy REAL,
            epoch INTEGER,
            duration INTEGER,
            {', '.join(index_colum)},         
        """ + ',\n'.join([f"FOREIGN KEY ({nm}) REFERENCES {nm} (name) ON DELETE CASCADE" for nm in dependent_tables]) + ')')

    # Add indexes for optimized reads
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accuracy_desc ON stat (accuracy DESC);")
    for nm in index_colum:
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{nm} ON stat ({nm});")

    # Create mobile analytics table (runtime stats)
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {mobile_table} (
        id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        device_type TEXT,
        os_version TEXT,
        valid BOOLEAN,
        emulator BOOLEAN,
        error_message TEXT,
        duration INTEGER,
        device_analytics_json TEXT,
        FOREIGN KEY (model_name) REFERENCES nn (name) ON DELETE CASCADE
    )
    """)
    # Indexes for mobile analytics
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{mobile_table}_model ON {mobile_table} (model_name);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{mobile_table}_device ON {mobile_table} (device_type);")
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
