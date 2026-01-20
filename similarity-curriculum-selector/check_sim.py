import sqlite3
import os

DB_PATH = "model_selection.db"

def check_sim():
    if not os.path.exists(DB_PATH):
        print("DB missing.")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM model_similarity")
    count = cur.fetchone()[0]
    print(f"Similarity Pairs (Edges): {count}")
    
    if count > 0:
        cur.execute("SELECT * FROM model_similarity LIMIT 5")
        print("Sample edges:", cur.fetchall())
        
    conn.close()

if __name__ == "__main__":
    check_sim()
