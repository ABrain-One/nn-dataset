import sqlite3
import json
import os
import re
from pathlib import Path

# Configuration
DB_PATH = "model_selection.db"
DATASETS_DIR = r"C:\Users\mailm\SQL\datasets\train"

def setup_database(conn):
    """Creates the necessary tables if they don't exist."""
    cur = conn.cursor()
    
    # Core Model Registry
    cur.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            task TEXT,
            dataset TEXT,
            metric TEXT,
            architecture TEXT,
            best_accuracy REAL,
            total_duration_ms INTEGER,
            path TEXT
        )
    """)
    
    # Similarity Index
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_similarity (
            source_model_id INTEGER,
            target_model_id INTEGER,
            similarity_score REAL,
            metric_type TEXT DEFAULT 'jaccard',
            PRIMARY KEY (source_model_id, target_model_id),
            FOREIGN KEY(source_model_id) REFERENCES models(id),
            FOREIGN KEY(target_model_id) REFERENCES models(id)
        )
    """)
    
    cur.execute("CREATE INDEX IF NOT EXISTS idx_similarity_score ON model_similarity(similarity_score)")
    conn.commit()

def parse_folder_name(folder_name):
    """
    Parses the folder name to extract metadata.
    Expected pattern: {task}_{dataset}_{metric}_{architecture}-{optional_hash}
    """
    # Heuristic parsing - split by underscores, assume order
    # This is brittle but a good starting point for this specific dataset structure
    parts = folder_name.split("_")
    
    if len(parts) >= 4:
        task = parts[0]
        dataset = parts[1]
        metric = parts[2]
        # Architecture might contain hyphens, so we take the rest and split off the hash if likely
        rest = "_".join(parts[3:])
        
        # Attempt to split architecture from hash/UUID
        # Typical UUIDs/hashes are long hex strings at the end
        # But some folders are just "ResNetTransformer" or "RESNETLSTM"
        
        # Regex for trailing hash/uuid: -[0-9a-f]{10,} or similar
        match = re.search(r"-(?:[0-9a-f]{8,}|[0-9]+)$", rest)
        if match:
            architecture = rest[:match.start()]
        else:
            architecture = rest
            
        return {
            "task": task,
            "dataset": dataset,
            "metric": metric,
            "architecture": architecture
        }
    
    return {
        "task": "unknown",
        "dataset": "unknown",
        "metric": "unknown",
        "architecture": folder_name
    }

def process_logs(folder_path):
    """
    Reads all .json log files in the folder and calculates aggregates.
    Returns best_accuracy, total_duration
    """
    best_acc = 0.0
    total_dur = 0
    
    try:
        # Scan for all JSON files
        json_files = list(Path(folder_path).glob("*.json"))
        
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Some files are lists of dicts, some might be single dicts
                    if isinstance(data, list):
                        records = data
                    else:
                        records = [data]
                        
                    for rec in records:
                        acc = rec.get("accuracy", 0.0)
                        dur = rec.get("duration", 0)
                        
                        if acc > best_acc:
                            best_acc = acc
                        total_dur += dur
                        
            except json.JSONDecodeError:
                pass # Skip corrupt files
            except Exception:
                pass
                
    except Exception as e:
        print(f"Error processing logs in {folder_path}: {e}")
        
    return best_acc, total_dur

def ingest_data():
    if not os.path.exists(DATASETS_DIR):
        print(f"Error: Dataset directory not found at {DATASETS_DIR}")
        return

    conn = sqlite3.connect(DB_PATH)
    setup_database(conn)
    cur = conn.cursor()
    
    print(f"Scanning directories in {DATASETS_DIR}...")
    
    # Use os.scandir for better performance with large directories
    with os.scandir(DATASETS_DIR) as it:
        # Filter for directories only
        folders = (entry for entry in it if entry.is_dir())
        
        params_list = []
        batch_size = 1000
        count = 0
        
        for folder in folders:
            count += 1
            if count % 100 == 0:
                print(f"Processing {count}...", end='\r')
            
            folder_name = folder.name
            folder_path = folder.path
        meta = parse_folder_name(folder_name)
        best_acc, total_dur = process_logs(folder)
        
        params_list.append((
            folder_name,
            meta["task"],
            meta["dataset"],
            meta["metric"],
            meta["architecture"],
            best_acc,
            total_dur,
            str(folder)
        ))
        
        if len(params_list) >= batch_size:
            cur.executemany("""
                INSERT OR IGNORE INTO models 
                (name, task, dataset, metric, architecture, best_accuracy, total_duration_ms, path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, params_list)
            conn.commit()
            params_list = []
            
    # Insert remaining
    if params_list:
        cur.executemany("""
            INSERT OR IGNORE INTO models 
            (name, task, dataset, metric, architecture, best_accuracy, total_duration_ms, path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, params_list)
        conn.commit()
        
    print("Ingestion complete.")
    
    # Verify count
    cur.execute("SELECT COUNT(*) FROM models")
    count = cur.fetchone()[0]
    print(f"Total models in database: {count}")
    
    conn.close()

if __name__ == "__main__":
    ingest_data()
