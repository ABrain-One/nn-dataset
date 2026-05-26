# Phase 1: Optuna search for MobileAgeNet on UTKFace age regression

import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    multiprocessing.freeze_support()

import ab.nn.util.db.Util as _db_util
_db_util.get_package_location = lambda x: None

from ab.nn.train import main
from ab.nn.util.Const import data_dir, db_file, code_tables
from ab.nn.util.db.Init import init_db, sql_conn, close_conn
from ab.nn.util.db.Write import populate_code_table
from ab.nn.util.db.Read import supported_transformers


def init_fresh_db(remove_existing: bool = True):
    print("Initializing database...")

    if remove_existing and db_file.exists():
        print(f"Removing old DB: {db_file}")
        db_file.unlink()

    init_db()
    conn, cursor = sql_conn()
    try:
        for table in code_tables:
            populate_code_table(table, cursor)
    finally:
        close_conn(conn)

    print("Database initialized with model code!\n")


def choose_transforms():
    all_transforms = supported_transformers()

    # Keep the search tight. UTKFace usually benefits more from stable crops/flip/norm
    # than from aggressive appearance distortion.
    preferred_transforms = [
        'norm_256_flip',
        'norm_256',
        'Resize_ColorJit_Flip_Blur',
        'bf-v1-RandomHorizontalFlip_RandomAutocontrast_256',
    ]

    valid_preferred = tuple(t for t in preferred_transforms if t in all_transforms)

    fallback_transforms = tuple(
        t for t in all_transforms
        if t not in ('five_crop',)
        and 'Grayscale' not in t
        and 'grayscale' not in t
        and 'Posterize' not in t
        and 'Solarize' not in t
    )

    face_transforms = valid_preferred if len(valid_preferred) >= 2 else fallback_transforms

    print(f"Selected {len(face_transforms)} transforms for search.")
    print(f"Transforms: {face_transforms}\n")
    return face_transforms


def maybe_upload_age_artifacts() -> None:
    """Optionally upload newly generated age artifacts to Hugging Face."""
    if os.environ.get("AGE_AUTO_UPLOAD_HF", "0") != "1":
        return

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("[WARN] AGE_AUTO_UPLOAD_HF=1 but HF_TOKEN is empty; skipping upload.")
        return

    script_path = Path(__file__).resolve().parent / "scripts" / "upload_age_artifacts_to_hf.py"
    if not script_path.exists():
        print(f"[WARN] Upload script not found: {script_path}")
        return

    cmd = [sys.executable, str(script_path), "--hf-token", hf_token]
    print("[INFO] Uploading age ONNX/TFLite artifacts to Hugging Face...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARN] Artifact upload exited with code {result.returncode}")


if __name__ == '__main__':
    RESET_DB = True

    NUM_WORKERS = int(os.environ.get("AGE_NUM_WORKERS", "8"))
    N_TRIALS = int(os.environ.get("AGE_OPTUNA_TRIALS", "40"))
    EPOCH_MAX = int(os.environ.get("AGE_EPOCH_MAX", "60"))

    nn_prm = {
        'freeze_epochs': 5,
        'backbone_lr_mult': 0.10,
        'bounded_output': 1,
        'min_age': 0.0,
        'max_age': 116.0,
        'train_min_age': 1.0,
        'train_max_age': 95.0,
    }

    init_fresh_db(remove_existing=RESET_DB)
    face_transforms = choose_transforms()

    config = 'age-regression_utkface_mae_MobileAgeNet'

    print("=" * 60)
    print("PHASE 1: HYPERPARAMETER SEARCH — MobileAgeNet on UTKFace")
    print("Target: MAE <= 3.5 years (acc >= 0.767 with threshold=15 years)")
    print("=" * 60)
    print(f"Config : {config}")
    print(f"Data   : {data_dir}")
    print(f"Trials : {N_TRIALS}")
    print(f"Epochs : {EPOCH_MAX}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Fixed model params: {nn_prm}")
    print("=" * 60)

    main(
        config=config,
        nn_prm=nn_prm,
        epoch_max=EPOCH_MAX,
        n_optuna_trials=N_TRIALS,

        min_batch_binary_power=6,
        max_batch_binary_power=7,

        min_learning_rate=5e-4,
        max_learning_rate=2e-3,

        # Framework compatibility only; AdamW ignores this in your model.
        min_momentum=0.90,
        max_momentum=0.90,

        min_dropout=0.10,
        max_dropout=0.30,

        transform=face_transforms,
        save_pth_weights=True,
        save_onnx_weights=1,
        num_workers=NUM_WORKERS,
        pretrained=1,
        epoch_limit_minutes=1060,
    )

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("Inspect DB, logs, TensorBoard, and saved weights for the best trial.")
    print("Then lock the best LR / batch / dropout / transform in train_final.py.")
    print("=" * 60)

    maybe_upload_age_artifacts()