"""
Train MobileAgeNet on UTKFace dataset using the existing pipeline.

This script uses the framework's training pipeline to train the age estimation model.
Configuration format: task_dataset_metric_model

Usage:
    python train_age_estimation.py
"""
import multiprocessing

# Fix multiprocessing on Windows
if __name__ == '__main__':
    multiprocessing.freeze_support()

from ab.nn.train import main
from ab.nn.util.Const import data_dir

if __name__ == '__main__':
    # Training configuration: task_dataset_metric_model
    config = 'age-regression_utkface_mae_MobileAgeNet'

    print("=" * 60)
    print("TRAINING MOBILEAGENET ON UTKFACE")
    print("=" * 60)
    print(f"\nConfig : {config}")
    print(f"Data   : {data_dir}")
    print("=" * 60)

    # Start training with Optuna optimization
    main(
        config=config,
        epoch_max=50,                    # More epochs for better convergence
        n_optuna_trials=30,              # More trials for better HP search
        min_batch_binary_power=5,        # 2^5 = 32
        max_batch_binary_power=7,        # 2^7 = 128 (cluster GPU handles larger batches)
        min_learning_rate=0.0001,        # Wider LR range for better search
        max_learning_rate=0.01,
        min_momentum=0.85,
        max_momentum=0.95,
        min_dropout=0.1,
        max_dropout=0.3,
        save_pth_weights=True,
        save_onnx_weights=1,
        num_workers=4,                   # Parallel data loading on Linux cluster
        epoch_limit_minutes=240,         # More time budget on cluster
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
