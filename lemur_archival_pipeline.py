"""
LEMUR DB Archival & Cleanup Pipeline
=====================================

Analyzes the nn-dataset repository to determine which Models, Statistics, and
Transforms are safe to archive/delete, cross-verifies them against the central
LEMUR database (db/ab.nn.db), generates an Excel audit report, compresses and
uploads the database to Hugging Face, and finally deletes local artifacts only
if every safety condition is satisfied.

Phases:
    0. Inventory Scan
    1. Dependency Mapping (Model <-> Stats <-> Transforms)
    2. SAFE / KEEP Logic
    3. DB Verification (hash + tuple lookups)
    4. Excel Audit Report
    5. Compress ab.nn.db -> ab.nn.db.zst
    6. Upload to Hugging Face
    7. Local Deletion (only SAFE + VERIFIED + UPLOAD OK)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from ab.nn.util.Const import ab_root_path, nn_dir, stat_train_dir, transform_dir, db_file

import json
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("lemur-pipeline")

# ---------------------------------------------------------------------------
# Paths / Constants
# ---------------------------------------------------------------------------

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
AUDIT_XLSX = ab_root_path / f"LEMUR_Deep_Audit_{TIMESTAMP}.xlsx"

# Reuse the project's existing HF/DB utilities instead of reimplementing them.
from ab.nn.util.Const import add_version  # noqa: E402
from ab.nn.util.ZST import compress as zst_compress  # noqa: E402
from ab.nn.util.hf.HF import upload_file as hf_upload_file  # noqa: E402
from ab.nn.util.hf.DB_from_HF import repo_id as HF_REPO_ID  # noqa: E402

# Versioned compressed filename, e.g. db/ab.nn.zst-2.2.9 (version comes from the
# project's `version` file via add_version()).
DB_ZST = db_file.parent / add_version("ab.nn.zst")

HF_TOKEN = os.environ.get("HF_TOKEN")  # or pass --HF_TOKEN on the command line


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class StatFolder:
    name: str  # e.g. "img-captioning_coco_bleu,meteor,cider_Blip2Fast"
    path: Path
    model_name: str
    task: str = ""
    dataset: str = ""
    nn: str = ""
    json_files: List[Path] = field(default_factory=list)
    transforms: Set[str] = field(default_factory=set)


@dataclass
class ModelEntry:
    name: str
    py_path: Path
    stat_folders: List[StatFolder] = field(default_factory=list)
    safe: bool = False


@dataclass
class TransformEntry:
    name: str
    py_path: Optional[Path]
    referenced_by: List[StatFolder] = field(default_factory=list)
    safe: bool = False


# ===========================================================================
# Phase 0: Inventory Scan
# ===========================================================================
def phase0_inventory() -> Tuple[Dict[str, ModelEntry], List[StatFolder], Dict[str, TransformEntry]]:
    """Scan the repository for Models, Statistic folders, and Transforms."""
    log.info("Phase 0: Inventory scan starting...")

    # --- Models -----------------------------------------------------------
    models: Dict[str, ModelEntry] = {}
    if nn_dir.exists():
        for py_file in nn_dir.glob("*.py"):
            if py_file.stem.startswith("__"):
                continue
            models[py_file.stem] = ModelEntry(name=py_file.stem, py_path=py_file)
    else:
        log.warning("Models directory not found: %s", nn_dir)

    # --- Statistic folders --------------------------------------------------
    stat_folders: List[StatFolder] = []
    if stat_train_dir.exists():
        for folder in stat_train_dir.iterdir():
            if not folder.is_dir():
                continue
            # Folder naming convention: <task>_<dataset>_<metric(s)>_<ModelName>
            parts = folder.name.split("_")
            if len(parts) < 4:
                log.warning("Unexpected stat folder naming convention: %s", folder.name)
                model_name = parts[-1] if parts else ""
                task = dataset = ""
            else:
                task = parts[0]
                dataset = parts[1]
                model_name = parts[-1]
                # everything between dataset and model name is the metric list
                # (not needed individually, kept for completeness)

            sf = StatFolder(
                name=folder.name,
                path=folder,
                model_name=model_name,
                task=task,
                dataset=dataset,
                nn=model_name,
            )
            sf.json_files = sorted(folder.glob("*.json"))
            stat_folders.append(sf)
    else:
        log.warning("Stats directory not found: %s", stat_train_dir)

    # --- Transforms ----------------------------------------------------------
    transforms: Dict[str, TransformEntry] = {}
    if transform_dir.exists():
        for py_file in transform_dir.glob("*.py"):
            if py_file.stem.startswith("__"):
                continue
            transforms[py_file.stem] = TransformEntry(name=py_file.stem, py_path=py_file)
    else:
        log.warning("Transforms directory not found: %s", transform_dir)

    log.info(
        "Phase 0 complete: %d models, %d stat folders, %d transforms",
        len(models), len(stat_folders), len(transforms),
    )
    return models, stat_folders, transforms


# ===========================================================================
# Phase 1: Dependency Mapping (Model <-> Stats <-> Transforms)
# ===========================================================================
def phase1_dependency_mapping(
    models: Dict[str, ModelEntry],
    stat_folders: List[StatFolder],
    transforms: Dict[str, TransformEntry],
) -> None:
    """Link models to their statistic folders, and statistic folders to transforms."""
    log.info("Phase 1: Dependency mapping starting...")

    # Map model -> stat folders
    for sf in stat_folders:
        model_entry = models.get(sf.model_name)
        if model_entry is not None:
            model_entry.stat_folders.append(sf)

        # Parse each JSON file for the "transform" key
        for jf in sf.json_files:
            try:
                with open(jf, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Failed to parse JSON %s: %s", jf, exc)
                continue

            transform_name = None
            if isinstance(data, dict):
                transform_name = data.get("transform")
            if not transform_name:
                continue

            sf.transforms.add(transform_name)

            t_entry = transforms.get(transform_name)
            if t_entry is None:
                # Transform referenced but no corresponding .py file found
                t_entry = TransformEntry(name=transform_name, py_path=None)
                transforms[transform_name] = t_entry
            t_entry.referenced_by.append(sf)

    log.info("Phase 1 complete.")


# ===========================================================================
# Phase 2: SAFE / KEEP Logic
# ===========================================================================
def phase2_safe_keep_logic(
    models: Dict[str, ModelEntry],
    transforms: Dict[str, TransformEntry],
) -> None:
    """
    Model <-> Stats:
        A model is SAFE TO PROCESS if it exists locally AND it has at least one
        local statistic folder AND every one of its statistic folders is eligible
        for archival (i.e. has at least one parsable JSON file referencing a
        transform).

    Stats <-> Transform:
        A transform is SAFE TO PROCESS only if every statistic folder that
        references it belongs to a model that is itself SAFE TO PROCESS.
    """
    log.info("Phase 2: SAFE/KEEP logic starting...")

    for model in models.values():
        if not model.py_path.exists():
            model.safe = False
            continue

        if not model.stat_folders:
            # No local statistics -> nothing to archive against, keep model
            model.safe = False
            continue

        all_stats_eligible = all(
            sf.path.exists() and len(sf.json_files) > 0 for sf in model.stat_folders
        )
        model.safe = all_stats_eligible

    for transform in transforms.values():
        if not transform.referenced_by:
            # No statistic folder references this transform locally -> keep
            transform.safe = False
            continue

        all_referencing_safe = all(
            models.get(sf.model_name) is not None and models[sf.model_name].safe
            for sf in transform.referenced_by
        )
        transform.safe = all_referencing_safe and transform.py_path is not None

    log.info("Phase 2 complete.")


# ===========================================================================
# Phase 3: DB Verification
# ===========================================================================
def _sha256_of_file(path: Path) -> str:
    """Return the SHA256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_lemur_dataframe() -> Optional[pd.DataFrame]:
    """Load the LEMUR DB dataframe via the internal API. Returns None on failure."""
    try:
        import ab.nn.api as api  # type: ignore
    except ImportError as exc:
        log.error("Failed to import 'ab.nn.api': %s", exc)
        return None

    try:
        df = api.data()
    except Exception as exc:  # pragma: no cover - defensive
        log.error("Failed to load dataframe via api.data(): %s", exc)
        return None

    log.info("Loaded LEMUR dataframe: %d rows, %d columns", len(df), len(df.columns))
    return df


def phase3_db_verification(
    models: Dict[str, ModelEntry],
    transforms: Dict[str, TransformEntry],
    df: Optional[pd.DataFrame],
) -> Dict[str, Dict[str, bool]]:
    """
    Verify SAFE candidates against the DB.

    Returns a dict keyed by artifact type ("model", "transform", "stat") mapping
    artifact-identifier -> bool (True = verified in DB).
    """
    log.info("Phase 3: DB verification starting...")

    verification: Dict[str, Dict[str, bool]] = {"model": {}, "transform": {}, "stat": {}}

    if df is None:
        log.warning("DB dataframe unavailable - all verifications will be marked Not Verified.")
        for model in models.values():
            if model.safe:
                verification["model"][model.name] = False
            for sf in model.stat_folders:
                verification["stat"][sf.name] = False
        for transform in transforms.values():
            if transform.safe:
                verification["transform"][transform.name] = False
        return verification

    # Pre-compute hash sets for nn_code and transform_code (drop NaNs)
    nn_code_hashes: Set[str] = set()
    if "nn_code" in df.columns:
        for code in df["nn_code"].dropna():
            nn_code_hashes.add(_sha256_of_text(str(code)))

    transform_code_hashes: Set[str] = set()
    if "transform_code" in df.columns:
        for code in df["transform_code"].dropna():
            transform_code_hashes.add(_sha256_of_text(str(code)))

    # Tuple set for (task, dataset, nn)
    tuple_set: Set[Tuple[str, str, str]] = set()
    if {"task", "dataset", "nn"}.issubset(df.columns):
        tuple_set = set(
            zip(df["task"].astype(str), df["dataset"].astype(str), df["nn"].astype(str))
        )

    # --- Model verification (hash of nn_code) -----------------------------
    for model in models.values():
        if not model.safe:
            continue
        try:
            local_hash = _sha256_of_file(model.py_path)
            verified = local_hash in nn_code_hashes
        except OSError as exc:
            log.warning("Could not read model file %s: %s", model.py_path, exc)
            verified = False
        verification["model"][model.name] = verified

    # --- Transform verification (hash of transform_code) ------------------
    for transform in transforms.values():
        if not transform.safe:
            continue
        if transform.py_path is None or not transform.py_path.exists():
            verification["transform"][transform.name] = False
            continue
        try:
            local_hash = _sha256_of_file(transform.py_path)
            verified = local_hash in transform_code_hashes
        except OSError as exc:
            log.warning("Could not read transform file %s: %s", transform.py_path, exc)
            verified = False
        verification["transform"][transform.name] = verified

    # --- Statistic verification (task, dataset, nn) tuple lookup ----------
    for model in models.values():
        for sf in model.stat_folders:
            if not model.safe:
                continue
            key = (sf.task, sf.dataset, sf.nn)
            verification["stat"][sf.name] = key in tuple_set

    log.info("Phase 3 complete.")
    return verification


# ===========================================================================
# Phase 4: Excel Audit Report
# ===========================================================================
def phase4_generate_audit(
    models: Dict[str, ModelEntry],
    transforms: Dict[str, TransformEntry],
    verification: Dict[str, Dict[str, bool]],
) -> bool:
    """Generate the multi-sheet Excel audit report. Returns True on success."""
    log.info("Phase 4: Generating Excel audit report -> %s", AUDIT_XLSX)

    try:
        # --- Sheet 1: Models & Stats --------------------------------------
        sheet1_rows = []
        for model in sorted(models.values(), key=lambda m: m.name):
            sheet1_rows.append({
                "Model Name": model.name,
                "Model Local": model.py_path.exists(),
                "Stat Folders Found": len(model.stat_folders),
                "Action": "SAFE TO PROCESS" if model.safe else "KEEP",
            })
        df_sheet1 = pd.DataFrame(
            sheet1_rows,
            columns=["Model Name", "Model Local", "Stat Folders Found", "Action"],
        )

        # --- Sheet 2: Transform Deep Dive ---------------------------------
        sheet2_rows = []
        for model in sorted(models.values(), key=lambda m: m.name):
            for sf in model.stat_folders:
                for jf in sf.json_files:
                    transform_name = ""
                    try:
                        with open(jf, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        if isinstance(data, dict):
                            transform_name = data.get("transform", "")
                    except (OSError, json.JSONDecodeError):
                        transform_name = "<unreadable>"

                    t_entry = transforms.get(transform_name)
                    action = "SAFE TO PROCESS" if (t_entry and t_entry.safe) else "KEEP"

                    sheet2_rows.append({
                        "Model Name": model.name,
                        "Statistic Folder": sf.name,
                        "JSON File": jf.name,
                        "Transform Name": transform_name,
                        "Action": action,
                    })
        df_sheet2 = pd.DataFrame(
            sheet2_rows,
            columns=["Model Name", "Statistic Folder", "JSON File", "Transform Name", "Action"],
        )

        # --- Sheet 3: DB Verification Audit -------------------------------
        sheet3_rows = []

        for model in sorted(models.values(), key=lambda m: m.name):
            safe_candidate = model.safe
            db_verified = verification["model"].get(model.name, False)
            final_action = "DELETE" if (safe_candidate and db_verified) else "KEEP"
            sheet3_rows.append({
                "Artifact Type": "Model",
                "Artifact Name": model.name,
                "Safe Candidate": "Yes" if safe_candidate else "No",
                "DB Verified": "Yes" if db_verified else "No",
                "Verification Method": "nn_code_hash",
                "Final Action": final_action,
            })

            for sf in model.stat_folders:
                stat_safe = model.safe
                stat_verified = verification["stat"].get(sf.name, False)
                stat_final = "DELETE" if (stat_safe and stat_verified) else "KEEP"
                sheet3_rows.append({
                    "Artifact Type": "Stat",
                    "Artifact Name": sf.name,
                    "Safe Candidate": "Yes" if stat_safe else "No",
                    "DB Verified": "Yes" if stat_verified else "No",
                    "Verification Method": "task+dataset+nn",
                    "Final Action": stat_final,
                })

        for transform in sorted(transforms.values(), key=lambda t: t.name):
            safe_candidate = transform.safe
            db_verified = verification["transform"].get(transform.name, False)
            final_action = "DELETE" if (safe_candidate and db_verified) else "KEEP"
            sheet3_rows.append({
                "Artifact Type": "Transform",
                "Artifact Name": transform.name,
                "Safe Candidate": "Yes" if safe_candidate else "No",
                "DB Verified": "Yes" if db_verified else "No",
                "Verification Method": "transform_code_hash",
                "Final Action": final_action,
            })

        df_sheet3 = pd.DataFrame(
            sheet3_rows,
            columns=[
                "Artifact Type", "Artifact Name", "Safe Candidate",
                "DB Verified", "Verification Method", "Final Action",
            ],
        )

        with pd.ExcelWriter(AUDIT_XLSX, engine="openpyxl") as writer:
            df_sheet1.to_excel(writer, sheet_name="Models & Stats", index=False)
            df_sheet2.to_excel(writer, sheet_name="Transform Deep Dive", index=False)
            df_sheet3.to_excel(writer, sheet_name="DB Verification Audit", index=False)

        log.info("Phase 4 complete: audit report written to %s", AUDIT_XLSX)
        return True

    except Exception as exc:  # pragma: no cover - defensive
        log.error("Failed to generate Excel audit report: %s", exc)
        return False


# ===========================================================================
# Phase 5: Compress ab.nn.db -> ab.nn.db.zst
# ===========================================================================
def phase5_compress_db() -> bool:
    """Compress db/ab.nn.db into the versioned db/ab.nn.zst-<version> file.

    Reuses the project's own `compress()` helper (zstandard, level 16, all
    cores) so behavior matches the rest of the codebase. The output filename
    already carries the project version via `add_version()`.
    """
    log.info("Phase 5: Compressing %s -> %s", db_file, DB_ZST)

    if not db_file.exists():
        log.error("Database file not found: %s", db_file)
        return False

    try:
        # remove=False: never delete the local source DB during compression;
        # actual deletion is governed solely by Phase 7's safety conditions.
        zst_compress(db_file, DB_ZST, False)
        log.info("Phase 5 complete: %s (%.2f MB)", DB_ZST, DB_ZST.stat().st_size / (1024 * 1024))
        return True
    except Exception as exc:  # pragma: no cover - defensive
        log.error("Failed to compress database: %s", exc)
        return False


# ===========================================================================
# Phase 6: Upload to Hugging Face
# ===========================================================================
def phase6_upload_to_hf(compression_ok: bool, audit_ok: bool, hf_token: Optional[str]) -> bool:
    """
    Upload the versioned compressed DB to the Hugging Face repo root, using
    the project's own `upload_file()` helper (which in turn uses
    `HfApi(token=...)` and `repo_type='model'`, creating the repo if needed).
    The Excel audit report is kept local only (not uploaded). Returns True
    only if the upload succeeds.
    """
    log.info("Phase 6: Uploading artifacts to Hugging Face repo '%s'...", HF_REPO_ID)

    if not (compression_ok and audit_ok):
        log.error("Skipping upload: prerequisite phases (compression/audit) failed.")
        return False

    token = hf_token or HF_TOKEN
    if not token:
        log.error(
            "No Hugging Face token found. Set the HF_TOKEN environment variable, "
            "pass --HF_TOKEN, or run 'huggingface-cli login'."
        )
        return False

    try:
        # remove=False: the local DB is only deleted by Phase 7's own
        # safety-gated cleanup, never as a side effect of uploading.
        # The audit report stays local only (not uploaded to HF) — see Phase 4.
        hf_upload_file(HF_REPO_ID, DB_ZST, DB_ZST, False, hf_token=token)
        log.info("Uploaded %s -> %s", DB_ZST.name, HF_REPO_ID)

        log.info("Phase 6 complete: upload succeeded.")
        return True

    except Exception as exc:  # pragma: no cover - defensive (network errors, auth, etc.)
        log.error("Hugging Face upload failed: %s", exc)
        return False


# ===========================================================================
# Phase 7: Local Deletion
# ===========================================================================
def phase7_delete_local(
    models: Dict[str, ModelEntry],
    transforms: Dict[str, TransformEntry],
    verification: Dict[str, Dict[str, bool]],
    compression_ok: bool,
    upload_ok: bool,
    audit_ok: bool,
) -> None:
    """
    Delete local artifacts ONLY if ALL conditions hold:
        1. Comparison logic marked it SAFE TO PROCESS.
        2. Artifact is Verified in the DB.
        3. DB compression succeeded.
        4. HF upload succeeded.
        5. Excel audit generated successfully.
    """
    log.info("Phase 7: Local deletion starting...")

    global_ok = compression_ok and upload_ok and audit_ok
    if not global_ok:
        log.warning(
            "Global safety conditions NOT met (compression=%s, upload=%s, audit=%s). "
            "No local artifacts will be deleted.",
            compression_ok, upload_ok, audit_ok,
        )
        return

    deleted_models, deleted_stats, deleted_transforms = 0, 0, 0

    # --- Delete model .py files -------------------------------------------
    for model in models.values():
        if model.safe and verification["model"].get(model.name, False):
            try:
                model.py_path.unlink(missing_ok=True)
                deleted_models += 1
                log.info("Deleted model file: %s", model.py_path)
            except OSError as exc:
                log.error("Failed to delete model %s: %s", model.py_path, exc)

            # --- Delete associated statistic folders -----------------------
            for sf in model.stat_folders:
                if verification["stat"].get(sf.name, False):
                    try:
                        for jf in sf.json_files:
                            jf.unlink(missing_ok=True)
                        # Remove the folder if now empty
                        if sf.path.exists() and not any(sf.path.iterdir()):
                            sf.path.rmdir()
                        deleted_stats += 1
                        log.info("Deleted statistic folder contents: %s", sf.path)
                    except OSError as exc:
                        log.error("Failed to delete stat folder %s: %s", sf.path, exc)
                else:
                    log.info("Keeping stat folder (not DB-verified): %s", sf.name)
        else:
            log.info("Keeping model (not SAFE/verified): %s", model.name)

    # --- Delete transform .py files ----------------------------------------
    for transform in transforms.values():
        if transform.safe and verification["transform"].get(transform.name, False):
            if transform.py_path and transform.py_path.exists():
                try:
                    transform.py_path.unlink()
                    deleted_transforms += 1
                    log.info("Deleted transform file: %s", transform.py_path)
                except OSError as exc:
                    log.error("Failed to delete transform %s: %s", transform.py_path, exc)
        else:
            log.info("Keeping transform (not SAFE/verified): %s", transform.name)

    log.info(
        "Phase 7 complete: %d models, %d stat folders, %d transforms deleted.",
        deleted_models, deleted_stats, deleted_transforms,
    )


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--HF_TOKEN', type=str, default=None)
    args = parser.parse_args()
    hf_token = args.HF_TOKEN or HF_TOKEN

    log.info("=== LEMUR DB Archival & Cleanup Pipeline starting ===")

    # Phase 0: Inventory Scan
    models, stat_folders, transforms = phase0_inventory()

    # Phase 1: Dependency Mapping
    phase1_dependency_mapping(models, stat_folders, transforms)

    # Phase 2: SAFE / KEEP Logic
    phase2_safe_keep_logic(models, transforms)

    # Phase 3: DB Verification
    df = load_lemur_dataframe()
    verification = phase3_db_verification(models, transforms, df)

    # Phase 4: Excel Audit Report (must run BEFORE any deletion)
    audit_ok = phase4_generate_audit(models, transforms, verification)

    # Phase 5: Compression
    compression_ok = phase5_compress_db()

    # Phase 6: Upload to Hugging Face
    upload_ok = phase6_upload_to_hf(compression_ok, audit_ok, hf_token)

    # Phase 7: Local Deletion (only if all conditions are satisfied)
    phase7_delete_local(
        models=models,
        transforms=transforms,
        verification=verification,
        compression_ok=compression_ok,
        upload_ok=upload_ok,
        audit_ok=audit_ok,
    )

    log.info("=== Pipeline finished ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
