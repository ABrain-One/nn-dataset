from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

from datasketch import MinHash, MinHashLSH
import numpy as np

DB_PATH = "db/ab.nn.db"
TABLE_SIG = "nn_code_minhash"
TABLE_SIM = "nn_similarity"

# Tunable pararms
LSH_THRESHOLD = 0.50   # candidate generation only
TOPK = 200             # store top-K neighbors per anchor
BATCH_INSERT = 5000    # rows per executemany
COMMIT_EVERY = 50000   # commit every N inserted edges


def minhash_from_hashvalues(hv: List[int], num_perm: int) -> MinHash:
    """
    Reconstruct a datasketch.MinHash from stored hashvalues.
    datasketch supports setting hashvalues directly.
    """
    m = MinHash(num_perm=num_perm)
    m.hashvalues = np.array(hv, dtype=np.uint64)  # REQUIRED for datasketch LSH
    return m

def minhash_jaccard_from_hashvalues(hv_a: List[int], hv_b: List[int]) -> float:
    if not hv_a or not hv_b:
        return 0.0
    if len(hv_a) != len(hv_b):
        return 0.0
    eq = 0
    for x, y in zip(hv_a, hv_b):
        if x == y:
            eq += 1
    return eq / len(hv_a)


@dataclass(frozen=True)
class SigRow:
    nn_name: str
    hv: List[int]
    num_perm: int
    shingle_n: int


def load_configs(conn: sqlite3.Connection) -> List[Tuple[int, int]]:
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT num_perm, shingle_n FROM {TABLE_SIG}")
    cfgs = [(int(a), int(b)) for a, b in cur.fetchall()]
    cfgs.sort()
    return cfgs


def load_signatures_for_config(conn: sqlite3.Connection, num_perm: int, shingle_n: int) -> List[SigRow]:
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT nn_name, hashvalues_json, num_perm, shingle_n
        FROM {TABLE_SIG}
        WHERE num_perm = ? AND shingle_n = ?
        """,
        (num_perm, shingle_n),
    )
    out: List[SigRow] = []
    for nn_name, hv_json, np_, sh_ in cur.fetchall():
        try:
            hv = json.loads(hv_json)
        except Exception:
            continue
        if not isinstance(hv, list) or len(hv) != int(np_):
            continue
        if not hv or not all(isinstance(x, int) for x in hv):
            continue
        out.append(SigRow(str(nn_name), hv, int(np_), int(sh_)))
    return out


def ensure_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_SIM} (
          nn_a TEXT NOT NULL,
          nn_b TEXT NOT NULL,
          jaccard REAL NOT NULL,
          num_perm INTEGER NOT NULL,
          shingle_n INTEGER NOT NULL,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (nn_a, nn_b)
        )
        """
    )
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_nn_similarity_a_j ON {TABLE_SIM} (nn_a, jaccard)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_nn_similarity_b ON {TABLE_SIM} (nn_b)")
    conn.commit()


def build_for_config(conn: sqlite3.Connection, rows: List[SigRow], num_perm: int, shingle_n: int) -> int:
    """
    Returns number of edges inserted.
    """
    if not rows:
        return 0

    print(f"[cfg num_perm={num_perm} shingle_n={shingle_n}] rows={len(rows)}")

    # Build MinHash objects + LSH index
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=num_perm)
    mh_by_name: Dict[str, MinHash] = {}
    hv_by_name: Dict[str, List[int]] = {}

    for r in rows:
        mh = minhash_from_hashvalues(r.hv, num_perm=r.num_perm)
        mh_by_name[r.nn_name] = mh
        hv_by_name[r.nn_name] = r.hv
        lsh.insert(r.nn_name, mh)

    cur = conn.cursor()

    inserted = 0
    buf: List[Tuple[str, str, float, int, int]] = []

    for i, a in enumerate(rows, 1):
        a_name = a.nn_name
        a_mh = mh_by_name[a_name]
        a_hv = hv_by_name[a_name]

        # LSH candidates
        cands = lsh.query(a_mh)
        if not cands:
            continue

        scored: List[Tuple[float, str]] = []
        for b_name in cands:
            if b_name == a_name:
                continue
            b_hv = hv_by_name.get(b_name)
            if b_hv is None:
                continue
            j = minhash_jaccard_from_hashvalues(a_hv, b_hv)
            scored.append((j, b_name))

        if not scored:
            continue

        scored.sort(reverse=True)  # by jaccard
        top = scored[:TOPK]

        for j, b_name in top:
            buf.append((a_name, b_name, float(j), int(num_perm), int(shingle_n)))

        # Flush
        if len(buf) >= BATCH_INSERT:
            cur.executemany(
                f"""
                INSERT INTO {TABLE_SIM} (nn_a, nn_b, jaccard, num_perm, shingle_n)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(nn_a, nn_b) DO UPDATE SET
                  jaccard=excluded.jaccard,
                  num_perm=excluded.num_perm,
                  shingle_n=excluded.shingle_n,
                  created_at=CURRENT_TIMESTAMP
                """,
                buf,
            )
            inserted += len(buf)
            buf.clear()

            if inserted % COMMIT_EVERY == 0:
                conn.commit()
                print(f"  inserted={inserted}")

        if i % 500 == 0:
            print(f"  processed {i}/{len(rows)} anchors")

    # Final flush
    if buf:
        cur.executemany(
            f"""
            INSERT INTO {TABLE_SIM} (nn_a, nn_b, jaccard, num_perm, shingle_n)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(nn_a, nn_b) DO UPDATE SET
              jaccard=excluded.jaccard,
              num_perm=excluded.num_perm,
              shingle_n=excluded.shingle_n,
              created_at=CURRENT_TIMESTAMP
            """,
            buf,
        )
        inserted += len(buf)
        buf.clear()

    conn.commit()
    print(f"[cfg done] inserted={inserted}")
    return inserted


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    ensure_table(conn)

    cfgs = load_configs(conn)
    print("configs:", cfgs)

    total = 0
    for num_perm, shingle_n in cfgs:
        rows = load_signatures_for_config(conn, num_perm, shingle_n)
        total += build_for_config(conn, rows, num_perm, shingle_n)

    print("TOTAL inserted edges:", total)
    conn.close()


if __name__ == "__main__":
    main()
