import os
import shutil
import time
import unittest
import uuid
from unittest.mock import patch

import pandas as pd

import ab.nn.api as api
import ab.nn.train as train
import ab.nn.util.db.Init as DB_Init
import ab.nn.util.db.Read as DB_Read
import ab.nn.util.db.Write as DB_Write
from ab.nn.api import JoinConf
from ab.nn.util.Const import *
from ab.nn.util.Util import read_py_file_as_string


class Testing(unittest.TestCase):
    inserted_uid: str | None = None  # set in setUpClass
    dummy_cfg_ext: tuple | None = None  # (task, dataset, metric, nn, epoch)

    @classmethod
    def setUpClass(cls):
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir)
            DB_Read.init_population()
        uid = str(uuid.uuid4())
        cls.inserted_uid = uid
        tmp_nn = f"tmp_{uid[:8]}"
        cls.dummy_cfg_ext = ("unit-task", "unit-set", "acc", tmp_nn, 1)

        dummy_code = "def model(x):\n    return x"
        prm = {
            "uid": uid,
            "lr": 0.123,
            "batch": 16,
            "note": "native-test",
            "flag": True,
            "transform": "norm_256_flip",
            "epoch": 1,
        }

        DB_Write.save_nn(
            nn_code=dummy_code,
            task="unit-task",
            dataset="unit-set",
            metric="acc",
            epoch=1,
            prm=prm,
            force_name=tmp_nn,
        )

    # ------------------------------------------------------------------
    #  Class-level tear-down: delete the synthetic record and confirm
    # ------------------------------------------------------------------
    @classmethod
    def tearDownClass(cls):
        if cls.inserted_uid is None:
            return

        conn, cur = DB_Init.sql_conn()
        cur.execute("DELETE FROM stat WHERE id=?", (cls.inserted_uid,))
        for tbl in param_tables:
            cur.execute(f"DELETE FROM {tbl} WHERE uid=?", (cls.inserted_uid,))

        # ── confirmation ───────────────────────────────────────────────
        cur.execute("SELECT COUNT(*) FROM stat WHERE id=?", (cls.inserted_uid,))
        assert cur.fetchone()[0] == 0, "stat row not removed in tearDownClass"

        for tbl in param_tables:
            cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE uid=?", (cls.inserted_uid,))
            leftovers = cur.fetchone()[0]
            assert leftovers == 0, f"{leftovers} leftover prm rows in {tbl}"

        DB_Init.close_conn(conn)

    # ------------------------------------------------------------------
    #  Basic data sanity check
    # ------------------------------------------------------------------
    def test_data(self):
        all_rows = DB_Read.data()
        img_rows = DB_Read.data(task="img-classification", nn=default_nn_name)
        img_3_rows = DB_Read.data(task="img-classification", nn=default_nn_name, max_rows=3, only_best_accuracy=True)
        print(img_3_rows)
        self.assertGreater(len(all_rows), len(img_rows))
        self.assertGreater(len(img_rows), len(img_3_rows))
        print(
            f"Total rows: {len(all_rows)} "
            f"(img-classification/{default_nn_name}: {len(img_rows)})"
        )

    # ------------------------------------------------------------------
    #  Basic data join check
    # ------------------------------------------------------------------
    def test_data_join(self):
        df = api.data(only_best_accuracy=True, task="img-classification",
                        nn_prefixes=('rag-',), max_rows=500,
                        sql=JoinConf(num_joint_nns=2,
                                     same_columns=('task', 'dataset', 'metric', 'epoch'),
                                     diff_columns=('nn',),
                                     enhance_nn=True))
        print(f"Total join rows: {len(df)} ")
        print(df.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    #  Basic 'run' data sanity check
    # ------------------------------------------------------------------
    def test_run_data(self):
        # Fetch first 10 rows from `run`
        df = api.run_data(max_rows=10)

        print("Total rows available:", len(api.run_data()))
        print("Showing first 10 rows:")
        print(df.to_string(index=False))

    # ------------------------------------------------------------------
    #  check_nn with trainer stubbed out
    # ------------------------------------------------------------------
    def test_check_nn(self):
        code = read_py_file_as_string(default_nn_path)

        result = api.check_nn(
            code,
            "img-classification",
            "cifar-10",
            "acc",
            {
                "lr": 0.01,
                "batch": 64,
                "dropout": 0.2,
                "momentum": 0.9,
                "transform": "norm_64_flip",
                "epoch": 1,
            },
            save_to_db=False,
        )
        print(f'Training results: {result}')

        with patch("ab.nn.util.Train.train_new") as mock_train:
            mock_train.return_value = ("mock-nn", 0.0, 0.0)
            result = api.check_nn(
                code,
                "img-classification",
                "cifar-10",
                "acc",
                {
                    "lr": 0.01,
                    "batch": 10,
                    "dropout": 0.2,
                    "momentum": 0.9,
                    "transform": "norm_256_flip",
                    "epoch": 1,
                },
                save_to_db=False,
            )

        mock_train.assert_called_once()
        self.assertEqual(result, ("mock-nn", 0.0, 0.0))

    # ------------------------------------------------------------------
    #  Native-typing round-trip
    # ------------------------------------------------------------------
    def test_native_param_types(self):
        rows = DB_Read.data(task="unit-task", nn=self.dummy_cfg_ext[3])
        self.assertEqual(len(rows), 1)
        prm = rows[0]["prm"]
        self.assertIsInstance(prm["lr"], float)
        self.assertIsInstance(prm["batch"], int)
        self.assertIsInstance(prm["note"], str)
        self.assertIn(prm["flag"], (0, 1))

    # ------------------------------------------------------------------
    #  Performance guard (dynamic budget)
    # ------------------------------------------------------------------
    def test_fetch_time_budget(self):
        start = time.perf_counter()
        rows = DB_Read.data()
        elapsed = time.perf_counter() - start
        print(f"DB_Read.data() fetched {len(rows)} rows in {elapsed:.3f}s")

        budget = max(10.0, 0.00025 * len(rows))  # 0.25 ms/row, ≥10 s
        self.assertLess(
            elapsed,
            budget,
            f"data() slower than expected ({elapsed:.2f}s > {budget:.2f}s)",
        )

    # ------------------------------------------------------------------
    #  remaining_trials logic
    # ------------------------------------------------------------------
    def test_remaining_trials(self):
        cfg = self.dummy_cfg_ext
        rem, passed = DB_Read.remaining_trials(cfg, 5)
        self.assertEqual(rem + passed, 5 if passed < 5 else passed)

        rem2, _ = DB_Read.remaining_trials(cfg, -3)
        self.assertEqual(rem2, 3)

        rem3, passed3 = DB_Read.remaining_trials(
            ("nope", "nope", "nope", "nope", 1), 7
        )
        self.assertEqual((rem3, passed3), (7, 0))

    # ------------------------------------------------------------------
    #  supported_transformers
    # ------------------------------------------------------------------
    def test_supported_transformers(self):
        tr = DB_Read.supported_transformers()
        self.assertIsInstance(tr, list)
        self.assertGreater(len(tr), 0)
        print("Transformers sample:", tr[:5])

    # ------------------------------------------------------------------
    #  unique_configs behaviour
    # ------------------------------------------------------------------
    def test_unique_configs(self):
        some = DB_Read.unique_configs(["img"])
        self.assertIsInstance(some, list)

        none = DB_Read.unique_configs(["nonexistent"])
        self.assertEqual(none, [])

    # ------------------------------------------------------------------
    #  DataFrame creation for a filtered query
    # ------------------------------------------------------------------
    def test_fetch_all_data_filters(self):
        df = pd.DataFrame(DB_Read.data(metric="acc"))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("metric", df.columns)

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------
    def test_nn_train(self):
        train.main('img-classification_cifar-10_acc_' + default_nn_name, {}, 1,
                   min_batch_binary_power=6, max_batch_binary_power=6, transform='norm_64_flip')


if __name__ == '__main__':
    unittest.main()
