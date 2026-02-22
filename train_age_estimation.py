# Train MobileAgeNet on UTKFace with Optuna hyperparameter optimization
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

import ab.nn.util.db.Util as _db_util
_db_util.get_package_location = lambda x: None

from ab.nn.train import main
from ab.nn.util.Const import data_dir, db_file, code_tables
from ab.nn.util.db.Init import init_db, sql_conn, close_conn
from ab.nn.util.db.Write import populate_code_table
from ab.nn.util.db.Read import supported_transformers

if __name__ == '__main__':
    # Initialize fresh database with model code
    print("Initializing database...")
    if db_file.exists():
        db_file.unlink()
    init_db()
    conn, cursor = sql_conn()
    for table in code_tables:
        populate_code_table(table, cursor)
    close_conn(conn)
    print("Database initialized with model code!\n")

    # Exclude broken five_crop transform (returns tuple instead of single image)
    all_transforms = supported_transformers()
    valid_transforms = tuple([t for t in all_transforms if t != 'five_crop'])
    print(f"Available transforms: {len(valid_transforms)} (excluded: five_crop)\n")

    config = 'age-regression_utkface_mae_MobileAgeNet'

    print("=" * 60)
    print("TRAINING MOBILEAGENET ON UTKFACE")
    print("=" * 60)
    print(f"\nConfig : {config}")
    print(f"Data   : {data_dir}")
    print("=" * 60)

    main(
        config=config,
        epoch_max=50,
        n_optuna_trials=30,
        min_batch_binary_power=5,
        max_batch_binary_power=7,
        min_learning_rate=0.0001,
        max_learning_rate=0.01,
        min_momentum=0.85,
        max_momentum=0.95,
        min_dropout=0.1,
        max_dropout=0.3,
        transform=valid_transforms,
        save_pth_weights=True,
        save_onnx_weights=1,
        num_workers=8,
        epoch_limit_minutes=240,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
