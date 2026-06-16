"""
Phase 2: Final MobileAgeNet training with locked best hyperparameters.
"""

import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    multiprocessing.freeze_support()

from ab.nn.train import main
from ab.nn.util.Const import data_dir


# Replace these after Phase 1.
BEST_LR = 0.0014162041314420217
BEST_BATCH_PW = 6                 # batch = 64
BEST_MOMENTUM = 0.90              # framework placeholder only
BEST_DROPOUT = 0.1807417434748134
BEST_TRANSFORM = ('Resize_ColorJit_Flip_Blur',)

BEST_FREEZE_EPOCHS = 5
BEST_BACKBONE_LR_MULT = 0.10
BEST_BOUNDED_OUTPUT = 1
BEST_MIN_AGE = 0.0
BEST_MAX_AGE = 116.0
BEST_TRAIN_MIN_AGE = 1.0
BEST_TRAIN_MAX_AGE = 95.0


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


def main_phase2():
    config = 'age-regression_utkface_mae_MobileAgeNet'

    num_workers = int(
        os.environ.get(
            "AGE_NUM_WORKERS",
            str(min(8, multiprocessing.cpu_count() if hasattr(multiprocessing, 'cpu_count') else 8))
        )
    )
    epoch_max = int(os.environ.get("AGE_FINAL_EPOCHS", "120"))
    epoch_limit_minutes = int(os.environ.get("AGE_FINAL_EPOCH_LIMIT_MIN", "1500"))

    nn_prm = {
        'freeze_epochs': BEST_FREEZE_EPOCHS,
        'backbone_lr_mult': BEST_BACKBONE_LR_MULT,
        'bounded_output': BEST_BOUNDED_OUTPUT,
        'min_age': BEST_MIN_AGE,
        'max_age': BEST_MAX_AGE,
        'train_min_age': BEST_TRAIN_MIN_AGE,
        'train_max_age': BEST_TRAIN_MAX_AGE,
    }

    print("=" * 60)
    print("PHASE 2: FINAL TRAINING WITH LOCKED HYPERPARAMETERS")
    print("Target: MAE <= 3.5 years (acc >= 0.767 with threshold=15 years)")
    print("=" * 60)
    print(f"Config              : {config}")
    print(f"Data                : {data_dir}")
    print(f"LR                  : {BEST_LR}")
    print(f"Batch               : 2^{BEST_BATCH_PW} = {2 ** BEST_BATCH_PW}")
    print(f"Momentum            : {BEST_MOMENTUM} (ignored by AdamW)")
    print(f"Dropout             : {BEST_DROPOUT}")
    print(f"Transform           : {BEST_TRANSFORM[0]}")
    print(f"Freeze epochs       : {BEST_FREEZE_EPOCHS}")
    print(f"Backbone LR mult    : {BEST_BACKBONE_LR_MULT}")
    print(f"Bounded output      : {BEST_BOUNDED_OUTPUT}")
    print(f"Age range           : [{BEST_MIN_AGE}, {BEST_MAX_AGE}]")
    print(f"Train target range  : [{BEST_TRAIN_MIN_AGE}, {BEST_TRAIN_MAX_AGE}]")
    print(f"Workers             : {num_workers}")
    print(f"Epochs              : {epoch_max}")
    print(f"Epoch limit (min)   : {epoch_limit_minutes}")
    print("=" * 60)

    main(
        config=config,
        nn_prm=nn_prm,
        epoch_max=epoch_max,
        n_optuna_trials=1,

        min_batch_binary_power=BEST_BATCH_PW,
        max_batch_binary_power=BEST_BATCH_PW,

        min_learning_rate=BEST_LR,
        max_learning_rate=BEST_LR,

        min_momentum=BEST_MOMENTUM,
        max_momentum=BEST_MOMENTUM,

        min_dropout=BEST_DROPOUT,
        max_dropout=BEST_DROPOUT,

        transform=BEST_TRANSFORM,
        save_pth_weights=True,
        save_onnx_weights=1,
        num_workers=num_workers,
        pretrained=1,
        epoch_limit_minutes=epoch_limit_minutes,
    )

    print("\n" + "=" * 60)
    print("PHASE 2 FINAL TRAINING COMPLETE")
    print("Review validation and held-out test MAE in the logs above.")
    print("=" * 60)

    maybe_upload_age_artifacts()


if __name__ == '__main__':
    main_phase2()