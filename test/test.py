import shutil
import time
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd

import ab.nn.api as api
import ab.nn.train as train
import ab.nn.util.db.Init as DB_Init
import ab.nn.util.db.Read as DB_Read
import ab.nn.util.db.Write as DB_Write
from ab.nn.api import JoinConf
from ab.nn.util.Const import *


# Models this run inserted, so tear-down can confirm each one was removed
# without touching anything it did not create.
created_models: list[str] = []


def drop_model(nm: str):
    """Remove one model and everything hanging off it.

    Nothing cascades here. `stat` declares FOREIGN KEY (prm) REFERENCES
    prm (uid), but `prm` has no unique key on `uid`, so switching foreign-key
    enforcement on makes SQLite reject every statement touching `stat` with
    "foreign key mismatch". The project therefore runs with enforcement off
    (`sql_conn` never sets the pragma) and each table is cleared explicitly,
    children first.
    """
    conn, cur = DB_Init.sql_conn()
    try:
        for tbl in param_tables:
            cur.execute(
                f"DELETE FROM {tbl} WHERE uid IN (SELECT prm FROM stat WHERE nn=?)",
                (nm,),
            )
        cur.execute("DELETE FROM stat WHERE nn=?", (nm,))
        cur.execute("DELETE FROM nn WHERE name=?", (nm,))
    finally:
        DB_Init.close_conn(conn)


def nn_row_count(nm: str) -> int:
    conn, cur = DB_Init.sql_conn()
    try:
        cur.execute("SELECT COUNT(*) FROM nn WHERE name=?", (nm,))
        return cur.fetchone()[0]
    finally:
        DB_Init.close_conn(conn)


@contextmanager
def dummy_nn():
    """Write a synthetic NN plus one stat row, yield it, then remove it.

    Yields the name `save_nn` reports, not the one requested: those differ
    whenever `code_to_db` recognises the architecture and files the statistics
    under the name already stored. Reading the return value is what keeps the
    rest of the block, and the deletion, pointed at the row that actually
    exists.

    The uid is embedded in the code so the architecture is new on every run and
    the requested name is the one that gets used. The row lives only for the
    duration of the `with` block; the deletion is in `finally`, so it runs when
    the body raises too.
    """
    uid = str(uuid.uuid4())
    requested = f"tmp_{uid[:8]}"
    code = f"# {uid}\ndef model(x):\n    return x"
    prm = {
        "uid": uid,
        "lr": 0.123,
        "batch": 16,
        "note": "native-test",
        "flag": True,
        "transform": "norm_256_flip",
        "epoch": 1,
    }

    nm = DB_Write.save_nn(
        nn_code=code,
        task="unit-task",
        dataset="unit-set",
        metric="acc",
        epoch=1,
        prm=prm,
        force_name=requested,
    )
    assert nm == requested, (
        f"save_nn filed the statistics under {nm!r}, not {requested!r}: "
        f"the synthetic architecture was already in the database"
    )
    created_models.append(nm)
    try:
        yield nm, uid, ("unit-task", "unit-set", "acc", nm, 1)
    finally:
        drop_model(nm)


class Testing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Rebuild the database, verify it, then test against it.

        The rebuild is unconditional: rebuilding only when `db_dir` already
        existed meant a first run on a clean checkout tested whatever
        `init_population()` had produced at import time. Cached query results
        from the previous database are dropped with it.
        """
        shutil.rmtree(db_dir, ignore_errors=True)
        DB_Write.init_population()
        assert db_file.exists(), f"database was not recreated at {db_file}"

        for fn in (api.data, api.run_data, api.prun_data,
                   api.nn_stat_data, api.layer_data):
            fn.cache_clear()

        assert DB_Read.data(max_rows=1), "database was recreated but is empty"

    @classmethod
    def tearDownClass(cls):
        """Every model this run inserted must be gone again."""
        leaked = [nm for nm in created_models if nn_row_count(nm)]
        assert not leaked, f"leaked models: {leaked}"

    # ------------------------------------------------------------------
    #  Basic data sanity check
    # ------------------------------------------------------------------
    def test_data(self):
        all_rows = DB_Read.data()
        img_rows = DB_Read.data(task="img-classification", nn=default_nn_name)
        img_3_rows = DB_Read.data(task="img-classification", nn=default_nn_name, max_rows=3, only_best_accuracy=True, include_nn_stats=True)
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
                      nn_prefixes=('rag-',), max_rows=500,  # , include_nn_stats=True # too slow
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
        df = api.run_data(max_rows=10)
        self.assertEqual(len(df), 10)
        print("Showing first 10 rows:")
        print(df.to_string(index=False))

        pt = api.run_data(type='pt')
        self.assertFalse(pt.empty)
        self.assertTrue((pt['type'] == 'pt').all())

    # ------------------------------------------------------------------
    #  check_nn with trainer stubbed out
    # ------------------------------------------------------------------
    def test_check_nn(self):
        code = DB_Read.nn_code(default_nn_name)

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
    #  save_nn round-trip: written, readable, and fully removed afterwards
    # ------------------------------------------------------------------
    def test_save_nn_roundtrip(self):
        with dummy_nn() as (nm, uid, _cfg):
            self.assertEqual(len(DB_Read.data(task="unit-task", nn=nm)), 1)
            self.assertEqual(nn_row_count(nm), 1)

        self.assertEqual(len(DB_Read.data(task="unit-task", nn=nm)), 0)
        self.assertEqual(nn_row_count(nm), 0)

        conn, cur = DB_Init.sql_conn()
        try:
            for tbl in param_tables:
                cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE uid=?", (uid,))
                self.assertEqual(cur.fetchone()[0], 0, f"leftover prm rows in {tbl}")
        finally:
            DB_Init.close_conn(conn)

    # ------------------------------------------------------------------
    #  Native-typing round-trip
    # ------------------------------------------------------------------
    def test_native_param_types(self):
        with dummy_nn() as (nm, _uid, _cfg):
            rows = DB_Read.data(task="unit-task", nn=nm)
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
        with dummy_nn() as (_nm, _uid, cfg):
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