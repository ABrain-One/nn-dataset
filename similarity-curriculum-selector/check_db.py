import sqlite3
import os

DB_PATH = "model_selection.db"

if not os.path.exists(DB_PATH):
    print("DB does not exist yet.")
else:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM models")
        count = cur.fetchone()[0]
        print(f"Models count: {count}")
        
        # Check first row
        cur.execute("SELECT * FROM models LIMIT 1")
        row = cur.fetchone()
        print(f"Sample row: {row}")
        conn.close()
    except Exception as e:
        print(f"Error reading DB: {e}")
