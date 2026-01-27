import sqlite3
import json
import re
import time
import hashlib
from pathlib import Path
import random

# Configuration
DB_PATH = "model_selection.db"

# --- Pure Python MinHash Implementation (Same as before) ---
MERSENNE_PRIME = (1 << 61) - 1
MAX_HASH = (1 << 32) - 1
HASH_RANGE = (1 << 32)

class SimpleMinHash:
    def __init__(self, num_perm=128, seed=1):
        self.num_perm = num_perm
        self.seed = seed
        self.max_val = MAX_HASH
        self.hashvalues = [self.max_val] * num_perm
        
        gen = random.Random(seed)
        self.perms = []
        for _ in range(num_perm):
            a = gen.randint(1, MERSENNE_PRIME - 1)
            b = gen.randint(0, MERSENNE_PRIME - 1)
            self.perms.append((a, b))

    def update(self, b_str):
        raw_hash = int(hashlib.sha1(b_str.encode('utf8')).hexdigest()[:8], 16)
        for i, (a, b) in enumerate(self.perms):
            ph = (a * raw_hash + b) % MERSENNE_PRIME
            ph = ph & MAX_HASH
            if ph < self.hashvalues[i]:
                self.hashvalues[i] = ph

    def jaccard(self, other):
        if not other: return 0.0
        matches = sum(1 for x, y in zip(self.hashvalues, other.hashvalues) if x == y)
        return matches / float(self.num_perm)


def compute_metadata_signature(model_row):
    """
    Constructs a signature from model metadata.
    row: (id, name, task, dataset, metric, architecture)
    """
    mid, name, task, dataset, metric, arch = model_row
    
    # Create a "virtual document" describing the model
    # We weight architecture heavily by adding it multiple times? 
    # Or just simple bag of tokens.
    
    # Token selection:
    # 1. Architecture (Critical)
    # 2. Dataset (Important context)
    # 3. Task (High level grouping)
    # 4. Metric (Optimization goal)
    
    tokens = [
        f"task:{task}", 
        f"dataset:{dataset}",
        f"metric:{metric}",
        f"arch:{arch}"
    ]
    
    # Split architecture into parts if it has hyphens (e.g. alt-nn1 -> alt, nn1)
    # to find partial matches? 
    # For now, keep it simple.
    
    mh = SimpleMinHash(num_perm=128)
    for t in tokens:
        mh.update(t)
    return mh

# ---------------------------------------------------------

def run_pipeline():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    print("Clearing old data...")
    cur.execute("DELETE FROM model_similarity WHERE metric_type = 'real_jaccard'")
    cur.execute("DELETE FROM model_similarity WHERE metric_type = 'synthetic_jaccard'")
    conn.commit()
    
    # 1. Fetch Models Metadata
    print("Fetching model metadata...")
    cur.execute("SELECT id, name, task, dataset, metric, architecture FROM models")
    rows = cur.fetchall()
    
    if not rows:
        print("No models found in DB.")
        return

    print(f"Processing {len(rows)} models for Metadata Similarity...")
    
    signatures = {} 
    
    # Step A: Compute Signatures
    sig_start = time.perf_counter()
    for i, row in enumerate(rows):
        model_id = row[0]
        mh = compute_metadata_signature(row)
        signatures[model_id] = mh
        
    print(f"Signatures computed in {time.perf_counter()-sig_start:.2f}s")
    
    # Step B: LSH Bucketing
    # Since we have highly repetitive metadata (many models have exact same task/dataset/arch),
    # collisions will be Frequent.
    # We want to link "Same Architecture, Different Hyperparams" (Wait, hyperparams are in JSON)
    # OR "Same Task, Different Architecture"
    
    # If we only used metadata in DB, exact duplicates will have Jaccard 1.0
    # The folders differ by Hyperparams (LR, Batch). 
    # But we didn't store those in DB 'models' table yet (only best_acc).
    # So basically, all "ResNet on Cifar10" models will look identical.
    # This creates a "Cluster" of identical models.
    # We can link them all (Dense Cluster) or just link a chain.
    
    print("Building LSH Buckets...")
    lsh_start = time.perf_counter()
    buckets = {} 
    
    # We use aggressive banding to split widely
    bands = 16 
    rows_per_band = 8
    
    for model_id, mh in signatures.items():
        for b in range(bands):
            start = b * rows_per_band
            end = start + rows_per_band
            # Create a tuple hash of this slice
            bucket_key = (b, tuple(mh.hashvalues[start:end]))
            
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(model_id)

    print(f"LSH Bucketing complete. {len(buckets)} buckets created.")
    print(f"LSH bucketing time: {time.perf_counter()-lsh_start:.2f}s")
    
    # Step C: Find Candidates
    print("Finding candidates and computing Jaccard...")
    edge_start = time.perf_counter()
    edges = set()
    match_count = 0
    candidate_computations = 0
    insert_sql = "INSERT OR IGNORE INTO model_similarity (source_model_id, target_model_id, similarity_score, metric_type) VALUES (?, ?, ?, 'meta_jaccard')"
    insert_batch = []
    batch_size = 1000
    
    # Optimization:
    # If a bucket has > 1000 items (e.g. all 5000 Cifar models), 
    # fully connecting them is 25 million edges.
    # We should only link a SAMPLE.
    
    MAX_LINKS_PER_MODEL = 20
    
    # Sort buckets by size to process small specific clusters first?
    # No, just iterate.
    
    for bucket_ids in buckets.values():
        if len(bucket_ids) < 2: continue
        
        # Sampling Strategy for large buckets
        candidates = bucket_ids
        if len(candidates) > 50:
             # Just pick random pairs?
             # Or verify similarity for a subset
             # We assume they are good candidates.
             pass

        # To avoid O(N^2) explosion in huge clusters:
        # For each member, pick k random partners from the same bucket.
        
        for id1 in candidates:
            # Pick random subset of others
            others = candidates[:]
            others.remove(id1)
            
            # Limit connections to avoid dense graph explosion
            if len(others) > 10:
                others = random.sample(others, 10)
                
            for id2 in others:
                pair = tuple(sorted((id1, id2)))
                if pair in edges: continue
                
                score = signatures[id1].jaccard(signatures[id2])
                candidate_computations += 1
                
                if score >= 0.5: 
                    edges.add(pair)
                    insert_batch.append((id1, id2, score))
                    # Single direction or dual? Selector logic often checks one way.
                    # curriculum generator does: join models t on target_id = t.id
                    # so if we are at src, we find next via target.
                    # Add dual?
                    # Let's save space and assume Undirected in logic? 
                    # No, SQL query is directional. Add both.
                    # Wait, saving 2x edges might blow up DB if dense.
                    # Let's just add (id1, id2). The graph is traversable.
                    
                    match_count += 1

                    if len(insert_batch) >= batch_size:
                        cur.executemany(insert_sql, insert_batch)
                        conn.commit()
                        insert_batch = []
        
        if match_count > 500000:
            print("Hit safety limit of 500k edges. Stopping.")
            break
            
        if match_count % 1000 == 0:
            print(f"Found {match_count} edges...", end='\r')

    if insert_batch:
        cur.executemany(insert_sql, insert_batch)

    conn.commit()
    conn.close()
    print(f"\nFinal: Found {match_count} metadata similarity edges.")
    print(f"Edge insertion time: {time.perf_counter()-edge_start:.2f}s")

if __name__ == "__main__":
    run_pipeline()
