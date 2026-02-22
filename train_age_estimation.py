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

# Force framework to use local writable path instead of read-only site-packages
import ab.nn.util.db.Util as _db_util
_db_util.get_package_location = lambda x: None

from ab.nn.train import main
from ab.nn.util.Const import data_dir, db_file, code_tables
from ab.nn.util.db.Init import init_db, sql_conn, close_conn
from ab.nn.util.db.Write import populate_code_table
from ab.nn.util.db.Read import supported_transformers

if __name__ == '__main__':
    # Ensure fresh database with model code
    print("Initializing database...")
    if db_file.exists():
        db_file.unlink()  # Delete old database
    init_db()  # Create tables
    
    # Populate model code into database
    conn, cursor = sql_conn()
    for table in code_tables:
        populate_code_table(table, cursor)
    close_conn(conn)
    print("Database initialized with model code!\n")
    
    # Get all available transforms and exclude broken ones
    # five_crop is broken: it returns a tuple of 5 images, but subsequent transforms expect a single image
    all_transforms = supported_transformers()
    valid_transforms = tuple([t for t in all_transforms if t != 'five_crop'])
    print(f"Available transforms: {len(valid_transforms)} (excluded: five_crop)\n")
    
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
        transform=valid_transforms,      # Use only valid transforms (exclude broken five_crop)
        save_pth_weights=True,
        save_onnx_weights=1,
        num_workers=4,                   # Parallel data loading on Linux cluster
        epoch_limit_minutes=240,         # More time budget on cluster
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
