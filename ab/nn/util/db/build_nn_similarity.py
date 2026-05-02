import struct
from typing import Optional
import sqlite3

NUM_PERM = 128
PACK_FMT = "<" + "I" * NUM_PERM
PACK_SIZE = struct.calcsize(PACK_FMT)  # 512 bytes

# SQLite UDF for on-the-fly Jaccard

def jaccard_blobs(blob_a: Optional[bytes], blob_b: Optional[bytes]) -> Optional[float]:
    """SQLite scalar UDF: jaccard_blobs(blob_a, blob_b) → REAL."""
    if blob_a is None or blob_b is None:
        return None
    hv_a = struct.unpack(PACK_FMT, blob_a)
    hv_b = struct.unpack(PACK_FMT, blob_b)
    eq = sum(1 for x, y in zip(hv_a, hv_b) if x == y)
    return eq / NUM_PERM

 #----Packing and Unpacking-----
def pack_hashvalues(hv: list[int]) -> bytes:
    """Serialize list[int] (len=128) → 512-byte BLOB."""
    if len(hv) != NUM_PERM:
        raise ValueError(f"Expected {NUM_PERM} hashvalues, got {len(hv)}")
    return struct.pack(PACK_FMT, *hv)


def unpack_hashvalues(blob: bytes) -> list[int]:
    """Deserialize 512-byte BLOB → list[int] (len=128)."""
    return list(struct.unpack(PACK_FMT, blob))

#----------Helpers---------

#This function is not in use in the current pipeline
def upsert_minhash(
    con: sqlite3.Connection,
    nn: str,
    hashvalues: list[int],
    num_perm: int = 128,
    shingle_n: int = 7,
) -> None:
    """Insert or replace a single nn's MinHash vector."""
    blob = pack_hashvalues(hashvalues)
    con.execute(
        """
        INSERT INTO nn_minhash(nn, hashvalues, num_perm, shingle_n)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(nn) DO UPDATE SET
            hashvalues = excluded.hashvalues,
            num_perm   = excluded.num_perm,
            shingle_n  = excluded.shingle_n,
            created_at = datetime('now')
        """,
        (nn, blob, num_perm, shingle_n),
    )
    con.commit()

def upsert_minhash_batch(
    con: sqlite3.Connection,
    records: list[dict],
) -> int:
    """
    Bulk upsert. Each record must have keys:
      nn (str), hashvalues (list[int]), num_perm (int), shingle_n (int).
    Returns number of rows processed.
    """
    rows = [
        (r["nn"], pack_hashvalues(r["hashvalues"]), r.get("num_perm", 128), r.get("shingle_n", 7))
        for r in records
    ]
    con.executemany(
        """
        INSERT INTO nn_minhash(nn, hashvalues, num_perm, shingle_n)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(nn) DO UPDATE SET
            hashvalues = excluded.hashvalues,
            num_perm   = excluded.num_perm,
            shingle_n  = excluded.shingle_n,
            created_at = datetime('now')
        """,
        rows,
    )
    con.commit()
    return len(rows)
