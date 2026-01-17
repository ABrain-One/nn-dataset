"""
NNAnalysisSimilarity (from per-model stats)

Build code-level similarity for architectures using the already-generated per-model
statistics in: ab/nn/stat/nn/*.json """

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("numpy is required (pip install numpy)") from e

try:
    from datasketch import MinHash, MinHashLSH
except Exception as e:
    raise RuntimeError("datasketch is required (pip install datasketch)") from e

# ---------- helpers ----------

def safe_float(x: Optional[float]) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def open_text_writer(path: Path):
    """
    Returns a file-like object opened for text writing
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")

@dataclass
class ModelSig:
    prm_id: str
    nn: str
    num_perm: int
    shingle_n: int
    mh: MinHash

def load_signatures_from_stats(stats_dir: Path) -> Tuple[List[ModelSig], Dict[str, str], Dict[str, int]]:
    """
    Load MinHash signatures from per-model stat JSONs.
    Returns: list of ModelSig, pid -> nn mapping, counters dict
    """
    files = sorted(stats_dir.glob("*.json"))
    sigs: List[ModelSig] = []
    pid2nn: Dict[str, str] = {}
    seen_pids = set()

    counters = {
        "total_files": len(files),
        "loaded": 0,
        "skipped_error_file": 0,
        "skipped_missing_prm_id": 0,
        "skipped_missing_code_minhash": 0,
        "skipped_unavailable": 0,
        "skipped_bad_hashvalues": 0,
        "skipped_duplicate_prm_id": 0,
        "skipped_inconsistent_params": 0,
    }


    # We expect num_perm/shingle_n to be consistent across the folder.
    expected_num_perm: Optional[int] = None
    expected_shingle_n: Optional[int] = None

    for fp in files:
        nn_name = fp.stem  # filename stem is the "architecture unit"
        try:
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            counters["skipped_error_file"] += 1
            continue

        if "error" in obj:
            counters["skipped_error_file"] += 1
            continue

        prm_id = obj.get("prm_id")

        if prm_id in seen_pids:
            counters["skipped_duplicate_prm_id"] += 1
            continue
        seen_pids.add(prm_id)

        if not prm_id:
            counters["skipped_missing_prm_id"] += 1
            continue
        prm_id = str(prm_id)

        cm = obj.get("code_minhash")
        if not isinstance(cm, dict):
            counters["skipped_missing_code_minhash"] += 1
            continue

        if int(cm.get("available", 0)) != 1:
            counters["skipped_unavailable"] += 1
            continue

        num_perm = cm.get("num_perm")
        shingle_n = cm.get("shingle_n")
        hv = cm.get("hashvalues")

        if not isinstance(num_perm, int) or not isinstance(shingle_n, int) or not isinstance(hv, list):
            counters["skipped_bad_hashvalues"] += 1
            continue

        if expected_num_perm is None:
            expected_num_perm = num_perm
        if expected_shingle_n is None:
            expected_shingle_n = shingle_n

        # Keep one consistent signature configuration across the whole dataset.
        if num_perm != expected_num_perm or shingle_n != expected_shingle_n:
            counters["skipped_inconsistent_params"] += 1
            continue

        if len(hv) != num_perm:
            counters["skipped_bad_hashvalues"] += 1
            continue

        try:
            hv_arr = np.array(hv, dtype=np.uint64)
        except Exception:
            counters["skipped_bad_hashvalues"] += 1
            continue

        # Reconstruct MinHash from stored hashvalues.
        mh = MinHash(num_perm=num_perm, hashvalues=hv_arr)

        sigs.append(ModelSig(prm_id=prm_id, nn=nn_name, num_perm=num_perm, shingle_n=shingle_n, mh=mh))
        pid2nn[prm_id] = nn_name
        counters["loaded"] += 1

    return sigs, pid2nn, counters

def build_lsh(sigs: List[ModelSig], threshold: float) -> MinHashLSH:
    if not sigs:
        raise RuntimeError("No signatures loaded. Nothing to index.")
    num_perm = sigs[0].num_perm
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for s in sigs:
        lsh.insert(s.prm_id, s.mh)
    return lsh

def compute_similarity(
    sigs: List[ModelSig],
    pid2nn: Dict[str, str],
    *,
    threshold: float,
    top_k: int,
) -> Iterable[Dict[str, Any]]:
    """
    Yields similarity records one-by-one
    """
    if not sigs:
        return

    num_perm = sigs[0].num_perm
    shingle_n = sigs[0].shingle_n

    lsh = build_lsh(sigs, threshold=threshold)
    pid2mh = {s.prm_id: s.mh for s in sigs}

    total = len(sigs)

    for idx, s in enumerate(sigs, start=1):
        pid = s.prm_id
        mh = s.mh
        nn_name = s.nn

        try:
            candidates = [c for c in lsh.query(mh) if c != pid]

            scored: List[Tuple[str, float]] = []
            for c in candidates:
                j = mh.jaccard(pid2mh[c])
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
                    "candidate_count": len(candidates),
                    "near_dup_count": sum(1 for j in top_js if j >= threshold),
                    "max_jaccard": max(top_js) if top_js else 0.0,
                    "mean_topk_jaccard": (sum(top_js) / len(top_js)) if top_js else 0.0,
                    "neighbors": [
                        {"prm_id": cid, "nn": pid2nn.get(cid, ""), "j": round(j, 4)}
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

        if idx % 250 == 0:
            print(f"[Similarity] {idx}/{total} processed", file=sys.stderr)

        yield rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_dir", type=str, default="ab/nn/stat/nn",
                    help="Directory with per-model stat JSONs (contains code_minhash.hashvalues).")
    ap.add_argument("--out", type=str, default="ab/nn/stat/nn_sim.jsonl.gz",
                    help="Output path. Use .jsonl or .jsonl.gz")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--top_k", type=int, default=25)
    args = ap.parse_args()

    stats_dir = Path(args.stats_dir)
    out_path = Path(args.out)

    if not stats_dir.exists():
        raise RuntimeError(f"stats_dir not found: {stats_dir}")

    print(f"[INFO] Loading per-model signatures from: {stats_dir}", file=sys.stderr)
    sigs, pid2nn, counters = load_signatures_from_stats(stats_dir)

    print("[INFO] Load summary:", file=sys.stderr)
    for k, v in counters.items():
        print(f"  - {k}: {v}", file=sys.stderr)

    if not sigs:
        raise RuntimeError("No valid signatures loaded. Check per-model stats generation.")

    print(f"[INFO] Using num_perm={sigs[0].num_perm}, shingle_n={sigs[0].shingle_n}", file=sys.stderr)
    print(f"[INFO] Building similarity: threshold={args.threshold}, top_k={args.top_k}", file=sys.stderr)

    ok = 0
    bad = 0

    with open_text_writer(out_path) as w:
        for rec in compute_similarity(
            sigs,
            pid2nn,
            threshold=args.threshold,
            top_k=args.top_k,
        ):
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if isinstance(rec.get("sim"), dict) and "error" in rec["sim"]:
                bad += 1
            else:
                ok += 1

    print(f"[DONE] Wrote: {out_path}", file=sys.stderr)
    print(f"[DONE] Records: {len(sigs)} (ok={ok}, fail={bad})", file=sys.stderr)


if __name__ == "__main__":
    main()

