"""
Phase 2: Train final MobileAgeNet with best hyperparameters from Optuna search.

Workflow:
  1. Run train_age_estimation.py  → Phase 1: 30 trials x 50 epochs (HP search)
  2. Run check_results.py         → See best HPs from Phase 1 logs
  3. Update the locked values below with the best HPs
  4. Run this script              → Phase 2: 1 trial x 100 epochs (final model)

Usage:
    python train_final.py
"""
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

from ab.nn.train import main
from ab.nn.util.Const import data_dir

# ── UPDATE THESE with the best HPs found in Phase 1 logs ──────────────────────
BEST_LR       = 0.001   # e.g. from: "Best trial lr=0.001"
BEST_BATCH_PW = 6       # e.g. 6 means batch=64
BEST_MOMENTUM = 0.90    # e.g. from: "Best trial momentum=0.90"
BEST_DROPOUT  = 0.20    # e.g. from: "Best trial dropout=0.20"
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = 'age-regression_utkface_mae_MobileAgeNet'

    print("=" * 60)
    print("PHASE 2: FINAL TRAINING WITH BEST HYPERPARAMETERS")
    print("=" * 60)
    print(f"\nConfig    : {config}")
    print(f"Data      : {data_dir}")
    print(f"LR        : {BEST_LR}")
    print(f"Batch     : 2^{BEST_BATCH_PW} = {2**BEST_BATCH_PW}")
    print(f"Momentum  : {BEST_MOMENTUM}")
    print(f"Dropout   : {BEST_DROPOUT}")
    print("=" * 60)

    main(
        config=config,
        epoch_max=100,
        n_optuna_trials=1,               # Single run — no HP search
        min_batch_binary_power=BEST_BATCH_PW,
        max_batch_binary_power=BEST_BATCH_PW,
        min_learning_rate=BEST_LR,
        max_learning_rate=BEST_LR,
        min_momentum=BEST_MOMENTUM,
        max_momentum=BEST_MOMENTUM,
        min_dropout=BEST_DROPOUT,
        max_dropout=BEST_DROPOUT,
        save_pth_weights=True,
        save_onnx_weights=1,
        num_workers=8,
        epoch_limit_minutes=480,         # 8 hours max
    )

    print("\n" + "=" * 60)
    print("FINAL TRAINING COMPLETE")
    print("Run: python check_results.py")
    print("=" * 60)
