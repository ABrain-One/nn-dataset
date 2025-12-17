"""NNAnalysisSimilarity.py
Compute code-level similarity metrics for all
LEMUR models using datasketch MinHash + LSH, and store results into stat_dir/nn.json
(merging into existing records by prm_id)."""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import re

# Optional Dependency import + feature flag

try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except Exception:
    _HAS_DATASKETCH = False

from ab.nn.util.Const import stat_dir

# Keep tokenizer/shingling consistent with nn_sftcodegen_rag.py in nn-gpt
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[^\s]")


#Tokenization function
def _tokenize(code: str) -> List[str]:
    return TOKEN_RE.findall(code or "")

#Shingles function
def _shingles(tokens: List[str], n: int = 7) -> List[str]:
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

#Covert code string to MinHash Signature
def to_minhash(code: str, num_perm: int = 128, n:int = 7) -> "MinHash":
    #Creates MinHash object
    mh = MinHash(num_perm=num_perm)
    #tokenizes code, creates shingles, loops through each shingle string
    for sh in _shingles(_tokenize(code), n=n):
        mh.update(sh.encode("utf-8"))
    return mh

#Avoid crashing if jaccard gives weird values.
def safe_float(x: Optional[float]) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return 0.0
        return float(x)
    except Exception:
        return 0.0

#Read existing JSON
def read_existing_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


#Creat index by prm_id, turns list of records into dict
def index_by_prm_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        pid = r.get("prm_id")
        if pid is None:
            continue
        out[str(pid)] = r
    return out

#Computing Similarity for All
def compute_similarity_for_all(
    df,
    *,
    id_col: str = "prm_id",
    name_col: str = "nn",
    code_col: str = "nn_code",
    threshold: float = 0.85,
    num_perm: int = 128,
    shingle_n: int = 7,
    top_k: int = 25,
) -> List[Dict[str, Any]]:

    if not _HAS_DATASKETCH:
        raise RuntimeError(
            "datasketch is not installed. Install it with: pip install datasketch"
        )
    # 1) Build MinHash for each model
    id2mh: Dict[str, MinHash] = {}
    id2name: Dict[str, str] = {}

    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), start = 1):
        pid = getattr(row,id_col)
        nn_name= getattr(row, name_col)
        nn_code = getattr(row, code_col)

        if pid is None or nn_code is None:
            continue
        pid_s = str(pid) #normalize id to string
        id2name[pid_s] = str(nn_name)

        try:
            id2mh[pid_s] = to_minhash(str(nn_code), num_perm=num_perm, n=shingle_n)
        except Exception as e:
            # If one model has weird encoding, still keep going
            # Store a None entry by skipping; later record will contain error.
            print(f"[WARN] MinHash failed for prm_id={pid_s}: {type(e).__name__}: {e}")

        #progess print
        if i % 500 == 0:
            print(f"[MinHash] {i}/{total} processed")

    keys = list(id2mh.keys())
    print(f"[INFO] Built MinHash for {len(keys)}/{total} models")

    # 2) Build global LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for k in keys:
        lsh.insert(k, id2mh[k])
    print(f"[INFO] LSH index built (threshold={threshold}, num_perm={num_perm})")

    # 3) Compute per-model similarity characteristics
    out: List[Dict[str, Any]] = []
    for idx, pid in enumerate(keys, start=1):
        mh = id2mh[pid]
        nn_name = id2name.get(pid, "")

        try:
            cands = [c for c in lsh.query(mh) if c != pid]
            scored: List[Tuple[str, float]] = []
            for c in cands:
                j = mh.jaccard(id2mh[c])
                scored.append((c, safe_float(j)))

            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:top_k]
            top_js = [j for _, j in top]

            rec = {
                "prm_id": pid,
                "nn": nn_name,
                "sim": {
                    "method": "minhash_lsh",
                    "threshold": threshold,
                    "num_perm": num_perm,
                    "shingle_n": shingle_n,
                    "top_k": top_k,
                    "candidate_count": len(cands),
                    "near_dup_count": sum(1 for j in top_js if j >= threshold),
                    "max_jaccard": max(top_js) if top_js else 0.0,
                    "mean_topk_jaccard": (sum(top_js) / len(top_js)) if top_js else 0.0,
                    "neighbors": [
                        {"prm_id": cid, "nn": id2name.get(cid, ""), "j": round(j, 4)}
                        for cid, j in top
                    ],
                },
            }
        except Exception as e:
            rec = {
                "prm_id": pid,
                "nn": nn_name,
                "sim": {
                    "method": "minhash_lsh",
                    "threshold": threshold,
                    "num_perm": num_perm,
                    "shingle_n": shingle_n,
                    "top_k": top_k,
                    "error": f"{type(e).__name__}: {e}",
                },
            }

        out.append(rec)

        if idx % 500 == 0:
            print(f"[Similarity] {idx}/{len(keys)} processed")

    return out


def merge_into_nn_json(
        base_rows: List[Dict[str, Any]],
        sim_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge sim_rows into base_rows by prm_id:
    - If prm_id exists, set base['sim'] = sim['sim']
    - If missing, append new record with prm_id/nn/sim
    """
    base_idx = index_by_prm_id(base_rows)

    for s in sim_rows:
        pid = str(s.get("prm_id"))
        if pid in base_idx:
            base_idx[pid]["sim"] = s.get("sim", {})
            # optionally also backfill nn name
            if "nn" not in base_idx[pid] and s.get("nn"):
                base_idx[pid]["nn"] = s["nn"]
        else:
            base_rows.append(s)

    return base_rows


def main():
    # -------- Phase 1: load dataset --------
    from ab.nn.api import data  # nn-dataset API

    df = data()

    # Keep only what we need (fast + safe)
    needed = ["nn", "nn_code", "prm_id"]
    df = df[needed].head(5000)

    # -------- Phase 2: configure similarity parameters --------
    threshold = 0.85
    num_perm = 128
    shingle_n = 7
    top_k = 25

    # Optional: dev limit to test quickly
    # df = df.head(500)

    print(f"[INFO] Models loaded: {len(df)}")
    print(f"[INFO] Params: threshold={threshold}, num_perm={num_perm}, shingle_n={shingle_n}, top_k={top_k}")

    # -------- Phase 3: compute similarity --------
    sim_rows = compute_similarity_for_all(
        df,
        threshold=threshold,
        num_perm=num_perm,
        shingle_n=shingle_n,
        top_k=top_k,
    )

    # -------- Phase 4: write / merge into nn.json --------
    out_path = stat_dir / "nn.json"
    base_rows = read_existing_json(out_path)
    merged = merge_into_nn_json(base_rows, sim_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    ok = sum(1 for r in sim_rows if "error" not in r.get("sim", {}))
    bad = len(sim_rows) - ok

    print(f"[DONE] Similarity computed for {len(sim_rows)} models (ok={ok}, fail={bad})")
    print(f"[DONE] Updated JSON → {out_path}")


if __name__ == "__main__":
    main()









