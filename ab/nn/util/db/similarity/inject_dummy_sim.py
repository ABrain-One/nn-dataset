import sqlite3
import random

DB_PATH = "model_selection.db"

def inject_dummy_similarity():
    """
    Since Mock LSH might have returned 0 matches, we inject some 
    dummy edges to verify the Selector and Curriculum logic.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Get all IDs
    cur.execute("SELECT id FROM models")
    ids = [r[0] for r in cur.fetchall()]
    
    if len(ids) < 2:
        print("Not enough models to link.")
        return

    print(f"Injecting synthetic edges for {len(ids)} models...")
    
    edges = []
    # Create a "random walk" web
    # Ensure some high similarity chains exist
    
    for i in range(len(ids)):
        src = ids[i]
        
        # Link to 3-5 random others
        targets = random.sample(ids, k=min(len(ids), 5))
        for tgt in targets:
            if src == tgt: continue
            
            # Random score concentrated around 0.5-0.9
            score = random.uniform(0.5, 0.99)
            edges.append((src, tgt, score, 'synthetic_jaccard'))
            # Undirected graph assumption: add reverse? 
            # The schema is directed (Source->Target), but Jaccard is symmetric.
            # We'll just add one way for now, Selector queries 'source_model_id'
            
    print(f"Generated {len(edges)} edges. Inserting...")
    
    cur.executemany("""
        INSERT OR IGNORE INTO model_similarity 
        (source_model_id, target_model_id, similarity_score, metric_type)
        VALUES (?, ?, ?, ?)
    """, edges)
    
    conn.commit()
    conn.close()
    print("Injection complete.")

if __name__ == "__main__":
    inject_dummy_similarity()
