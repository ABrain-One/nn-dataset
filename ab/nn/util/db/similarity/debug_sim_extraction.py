import sqlite3
import json
import re
from pathlib import Path
from compute_similarity import extract_python_block, compute_signature

DB_PATH = "model_selection.db"

def debug_single_model():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Get a random model
    cur.execute("SELECT id, path FROM models ORDER BY RANDOM() LIMIT 1")
    row = cur.fetchone()
    if not row:
        print("No models.")
        return
        
    mid, folder_path = row
    print(f"Inspecting Model {mid}: {folder_path}")
    
    json_files = list(Path(folder_path).glob("*.json"))
    if not json_files:
        print("No JSON files found.")
        return

    with open(json_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Raw Data Type: {type(data)}")
    
    code_content = ""
    # Mirror the logic from script
    if isinstance(data, list) and data:
        print("Data is a List.")
        first = data[0]
        msgs = first.get("messages", [])
    elif isinstance(data, dict):
        print("Data is a Dict.")
        msgs = data.get("messages", [])
    else:
        print("Unknown format.")
        msgs = []

    print(f"Messages found: {len(msgs)}")
    for m in msgs:
        print(f"- Role: {m.get('role')}")
        if m.get("role") == "assistant":
            code_content = m.get("content", "")
            print(f"  -> Content Length: {len(code_content)}")
            # Print snippet
            print(f"  -> Snippet: {code_content[:100]}...")

    if not code_content:
        print("FAIL: No code content extracted.")
        return

    clean_code = extract_python_block(code_content)
    print(f"\nExtracted Code Block ({len(clean_code)} chars):")
    print("-" * 20)
    print(clean_code[:200])
    print("-" * 20)
    
    mh = compute_signature(clean_code)
    print(f"MinHash Generated. Values[0:5]: {mh.hashvalues[:5]}")
    
    # Check if infinity
    num_inf = sum(1 for x in mh.hashvalues if x == mh.max_val)
    print(f"Infinity Count: {num_inf} / 128")
    
    if num_inf == 128:
        print("ERROR: Hash signature is empty (all infinities). Tokenization failed?")

if __name__ == "__main__":
    debug_single_model()
