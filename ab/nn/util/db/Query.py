from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from sqlite3 import Cursor
from typing import Optional, Iterable

from ab.nn.util.Const import main_columns_ext, tmp_data


@dataclass(frozen=True)
class JoinConf:
    """
    Phase-1 query configuration:
      - SQL does functional equivalence (filtering slice via Read.data filters)
      - Python enforces architecture diversity using code_minhash signatures
    """

    # Required
    num_joint_nns: int

    # Optional: legacy knobs (kept for compatibility)
    same_columns: Optional[tuple[str, ...]] = None
    diff_columns: Optional[tuple[str, ...]] = None
    enhance_nn: Optional[bool] = None  # if True: prefer accuracy improvements (best-first anyway)

    # NEW: architecture diversity knobs
    use_arch_diversity: bool = False
    max_arch_jaccard: float = 0.85  # keeps candidates whose MinHash-Jaccard <= this vs already selected
    min_arch_jaccard: float = 0.0
    arch_stat_dir: str = "ab/nn/stat/nn"  # where per-model JSON stats live
    overfetch_factor: int = 20  # fetch more candidates then filter in python
    allow_fallback_fill: bool = True  # if not enough diverse models found, fill remaining by accuracy


    supported_columns = main_columns_ext

    def validate(self):
        if self.num_joint_nns < 1:
            raise ValueError("'num_joint_nns' must be >= 1.")

        if self.diff_columns:
            for c in self.diff_columns:
                if c not in self.supported_columns:
                    raise ValueError(f"Unsupported column name in diff_columns: {c}")

        if self.same_columns:
            for c in self.same_columns:
                if c not in self.supported_columns:
                    raise ValueError(f"Unsupported column name in same_columns: {c}")

        if self.enhance_nn is not None and not isinstance(self.enhance_nn, bool):
            raise ValueError("'enhance_nn' must be boolean.")

        if not (0.0 <= float(self.max_arch_jaccard) <= 1.0):
            raise ValueError("'max_arch_jaccard' must be within [0, 1].")

        if int(self.overfetch_factor) < 1:
            raise ValueError("'overfetch_factor' must be >= 1.")
        if self.use_arch_diversity:
            mn = float(getattr(self, "min_arch_jaccard", 0.0) or 0.0)
            mx = float(self.max_arch_jaccard)

            if not (0.0 <= mn <= mx <= 1.0):
                raise ValueError(f"Invalid arch band: min_arch_jaccard={mn}, max_arch_jaccard={mx}")


def join_nn_query(sql: JoinConf, limit_clause: Optional[str], cur: Cursor):
    """
    Reads from temp table tmp_data created by Read.data().
    Returns list[dict] (same shape as fill_hyper_prm()).

    Steps:
      1) Get candidates ordered by accuracy DESC
      2) Optionally filter by architecture diversity using per-model MinHash signatures
      3) Attach diversity explanation (pairwise summary) + optional arch summaries
    """
    sql.validate()
    n = int(sql.num_joint_nns)

    require_diff_nn = bool(sql.diff_columns) and ("nn" in sql.diff_columns)
    cand_n = max(n, 1) * int(sql.overfetch_factor)

    # Candidate selection
    if require_diff_nn:
        cur.execute(
            f"""
            WITH base AS (
              SELECT * FROM {tmp_data}
            ),
            best_per_nn AS (
              SELECT
                b.*,
                ROW_NUMBER() OVER (PARTITION BY b.nn ORDER BY b.accuracy DESC) AS rn
              FROM base b
            ),
            unique_nn AS (
              SELECT * FROM best_per_nn WHERE rn = 1
            )
            SELECT * FROM unique_nn
            ORDER BY accuracy DESC, nn ASC, epoch ASC
            LIMIT ?
            """,
            (cand_n,),
        )
    else:
        cur.execute(
            f"""
            SELECT *
            FROM {tmp_data}
            ORDER BY accuracy DESC, nn ASC, epoch ASC
            LIMIT ?
            """,
            (cand_n,),
        )

    candidates = fill_hyper_prm(cur, num_joint_nns=1)


#---------------Architecture Diversity-------------------
    # Fast path: no arch diversity, just top-N
    if not sql.use_arch_diversity:
        picked = candidates[:n]
        # optional: attach summaries if available
        attach_arch_summaries(picked)
        return picked

    stat_dir = Path(sql.arch_stat_dir)

    # Architecture diversity filter in Python (Option 1)
    selected = select_diverse_by_minhash(
        candidates,
        k=n,
        min_jaccard=float(sql.min_arch_jaccard),
        max_jaccard=float(sql.max_arch_jaccard),
        stat_dir=stat_dir,
        allow_fallback_fill=bool(sql.allow_fallback_fill),
    )

    # Attach optional nn_stat-based summaries (if present)
    attach_arch_summaries(selected)

    # how diverse is the final set
    sel_summary = pairwise_minhash_summary(selected, stat_dir=stat_dir)

    # Put it inside diversity_meta
    if selected:
        selected[0].setdefault("diversity_meta", {})
        if isinstance(selected[0]["diversity_meta"], dict):
            selected[0]["diversity_meta"]["pairwise_summary"] = sel_summary

    return selected



def fill_hyper_prm(cur: Cursor, num_joint_nns=1, include_nn_stats=False) -> list[dict]:
    rows = cur.fetchall()
    if not rows:
        return []
    columns = [c[0] for c in cur.description]

    from collections import defaultdict
    prm_by_uid: dict[str, dict[str, int | float | str]] = defaultdict(dict)

    cur.execute("SELECT uid, name, value FROM prm")
    for uid, name, value in cur.fetchall():
        prm_by_uid[uid][name] = value

    results: list[dict] = []
    for r in rows:
        rec = dict(zip(columns, r))
        rec["prm"] = prm_by_uid.get(rec["prm_id"], {})
        for i in range(2, num_joint_nns + 1):
            s = str(i)
            rec["prm_" + s] = prm_by_uid.get(rec["prm_id_" + s], {})
        rec.pop("transform", None)

        if include_nn_stats and "nn_stats_meta" in rec and rec["nn_stats_meta"]:
            try:
                rec["nn_stats_meta"] = json.loads(rec["nn_stats_meta"])
            except Exception:
                rec["nn_stats_meta"] = None

        results.append(rec)
    return results


# -----------------------------
# helpers - MinHash diversity
# -----------------------------

@lru_cache(maxsize=500000)
def _load_code_minhash_signature(stat_json_path: str) -> dict:
    """
    Cached JSON loader for per-model stat file.
    """
    with open(stat_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_hashvalues(nn_name: str, stat_dir: Path):
    """
    Returns (hashvalues_list, num_perm, shingle_n) or None if unavailable.
    """
    p = stat_dir / f"{nn_name}.json"
    if not p.exists():
        return None

    try:
        j = _load_code_minhash_signature(str(p))
    except Exception:
        return None

    if not isinstance(j, dict):
        return None

    if "error" in j:
        return None

    cm = j.get("code_minhash")
    if not isinstance(cm, dict):
        return None

    if int(cm.get("available", 0)) != 1:
        return None

    hv = cm.get("hashvalues")
    if not isinstance(hv, list) or not hv:
        return None

    # sanity: hv should be ints
    if not all(isinstance(x, int) for x in hv):
        return None

    num_perm = int(cm.get("num_perm", len(hv)))
    shingle_n = int(cm.get("shingle_n", 7))
    if len(hv) != num_perm:
        # inconsistent signature
        return None

    return hv, num_perm, shingle_n


def minhash_jaccard_from_hashvalues(hv_a: list[int], hv_b: list[int]) -> float:
    """
    MinHash Jaccard estimate from signature arrays:
    J ≈ (# positions equal) / num_perm
    """
    if not hv_a or not hv_b:
        return 0.0
    if len(hv_a) != len(hv_b):
        return 0.0
    eq = 0 # position equal
    for x, y in zip(hv_a, hv_b):
        if x == y:
            eq += 1
    return eq / len(hv_a)

def select_diverse_by_minhash(
    candidates: list[dict],
    *,
    k: int,
    min_jaccard: float = 0.0,
    max_jaccard: float = 1.0,
    stat_dir: Path,
    allow_fallback_fill: bool,
    restrict_fallback_to_max: bool = True,
) -> list[dict]:
    """
    Greedy selection:
      - candidates assumed sorted by accuracy DESC
      - pick next candidate only if its MinHash-Jaccard <= max_jaccard vs ALL selected

    If allow_fallback_fill=True, fill remaining slots by accuracy even if diversity constraint blocks.
    """
    # Similarity band
    min_j = float(min_jaccard)
    max_j = float(max_jaccard)

    if not (0.0 <= min_j <= max_j <= 1.0):
        raise ValueError(f"Invalid arch band: min_jaccard={min_j}, max_jaccard={max_j}")

    selected: list[dict] = []
    selected_sigs: list[list[int]] = []

    meta = {
        "use_arch_diversity": True,
        "min_arch_jaccard": min_j,
        "max_arch_jaccard": max_j,
        "arch_stat_dir": str(stat_dir),
        "requested_k": int(k),
        "selected_k": 0,
        "skipped_missing_signature": 0,
        "skipped_too_similar": 0,
        "skipped_too_dissimilar": 0,
        "fallback_filled": 0,
        "restrict_fallback_to_max": bool(restrict_fallback_to_max),
        "selection_rule": "band: accept first; later require (all j<=max) and (max(j)>=min if min>0)",
    }

    def _sig_for(rec: dict) -> Optional[list[int]]:
        nn_name = rec.get("nn")
        if not isinstance(nn_name, str) or not nn_name:
            return None
        sig_info = _extract_hashvalues(nn_name, stat_dir)
        if sig_info is None:
            return None
        hv, _, _ = sig_info
        return hv

    def _all_js(hv: list[int]) -> list[float]:
        return [minhash_jaccard_from_hashvalues(hv, hv_sel) for hv_sel in selected_sigs]

    for rec in candidates:
        if len(selected) >= k:
            break

        hv = _sig_for(rec)
        if hv is None:
            meta["skipped_missing_signature"] += 1
            continue

        # anchor
        if not selected_sigs:
            selected.append(rec)
            selected_sigs.append(hv)
            continue

        js = _all_js(hv)

        # Rule 1: not too similar to any selected
        if any(j > max_j for j in js):
            meta["skipped_too_similar"] += 1
            continue

        # Rule 2: if min enabled, must be similar to at least one selected
        if min_j > 0.0 and (max(js) < min_j):
            meta["skipped_too_dissimilar"] += 1
            continue
        selected.append(rec)
        selected_sigs.append(hv)

    #enforces "no near-duplicates"
    if allow_fallback_fill and len(selected) < k:
        already = {r.get("id") for r in selected}
        for rec in candidates:
            if len(selected) >= k:
                break
            if rec.get("id") in already:
                continue

            if restrict_fallback_to_max:
                hv = _sig_for(rec)
                if hv is None:
                    meta["skipped_missing_signature"] += 1
                    continue
                js = _all_js(hv) if selected_sigs else []
                if js and any(j > max_j for j in js):
                    meta["skipped_too_similar"] += 1
                    continue

                selected_sigs.append(hv)

            selected.append(rec)
            meta["fallback_filled"] += 1

    meta["selected_k"] = len(selected)

    # attach meta
    for r in selected:
        r.setdefault("diversity_meta", meta)
    return selected

from itertools import combinations
from typing import Any

def pairwise_minhash_summary(
    selected: list[dict],
    *,
    stat_dir: Path,
) -> dict[str, Any]:
    """
    For the final selected K models:
    - load their code_minhash.hashvalues (from per-model stat files)
    - The selector is written, so signature retrieval can be moved to DB later without changing selection logic
    - compute pairwise MinHash-Jaccard
    - return a compact summary that explains "how diverse" the set really is
    """
    # Load signatures once
    sigs: list[tuple[str, list[int]]] = []
    missing = 0

    for r in selected:
        nn_name = r.get("nn")
        if not isinstance(nn_name, str) or not nn_name:
            missing += 1
            continue

        sig_info = _extract_hashvalues(nn_name, stat_dir)
        if sig_info is None:
            missing += 1
            continue

        hv, _, _ = sig_info
        sigs.append((nn_name, hv))

    out: dict[str, Any] = {
        "k": len(selected),
        "with_signature": len(sigs),
        "missing_signature": missing,
        "pairwise_max_j": None,
        "pairwise_mean_j": None,
        "closest_pair": None,     # most similar (highest J)
        "pairs_top": [],          # top few closest pairs
    }

    if len(sigs) < 2:
        return out

    pairs = []
    vals = []

    for (nn_a, hv_a), (nn_b, hv_b) in combinations(sigs, 2):
        j = minhash_jaccard_from_hashvalues(hv_a, hv_b)
        vals.append(j)
        pairs.append({"a": nn_a, "b": nn_b, "j": round(j, 4)})

    pairs.sort(key=lambda x: x["j"], reverse=True)

    out["pairwise_max_j"] = round(max(vals), 4) if vals else None
    out["pairwise_mean_j"] = round(sum(vals) / len(vals), 4) if vals else None
    out["closest_pair"] = pairs[0] if pairs else None
    out["pairs_top"] = pairs[:10]  # keep it small

    return out


def attach_arch_summaries(selected: list[dict]) -> None:
    """
    Adds `architecture summary` if nn_stat fields are present in each row.
    This is optional and only works if Read.data(include_nn_stats=True).
    """
    for r in selected:
        # only if nn_stat columns exist
        if "nn_total_params" not in r:
            r["arch_summary"] = {}
            continue

        r["arch_summary"] = {
            "total_params": r.get("nn_total_params"),
            "trainable_params": r.get("nn_trainable_params"),
            "total_layers": r.get("nn_total_layers"),
            "leaf_layers": r.get("nn_leaf_layers"),
            "max_depth": r.get("nn_max_depth"),
            "flops": r.get("nn_flops"),
            "model_size_mb": r.get("nn_model_size_mb"),
            "dropout_count": r.get("nn_dropout_count"),
            "has_attention": r.get("nn_has_attention"),
            "has_residual": r.get("nn_has_residual"),
            "is_resnet_like": r.get("nn_is_resnet_like"),
            "is_transformer_like": r.get("nn_is_transformer_like"),
        }
