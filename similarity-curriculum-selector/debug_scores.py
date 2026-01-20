import sqlite3
import statistics

conn = sqlite3.connect("model_selection.db")
cur = conn.cursor()

cur.execute("SELECT similarity_score FROM model_similarity LIMIT 1000")
scores = [r[0] for r in cur.fetchall()]

if not scores:
    print("No similarity scores found.")
else:
    print(f"Count: {len(scores)}")
    print(f"Min: {min(scores)}")
    print(f"Max: {max(scores)}")
    print(f"Avg: {statistics.mean(scores)}")
    print(f"Sample: {scores[:10]}")

conn.close()
