import unittest
import uuid

import ab.nn.util.db.Init as DB_Init
import ab.nn.util.db.Read as DB_Read
import ab.nn.util.db.Write as DB_Write
from ab.nn.util.db.Query import JoinConf


class SqlQueryPathTests(unittest.TestCase):
    dataset = "unit-sql-set"
    metric = "acc"
    epoch = 1
    transform = "unit_transform"
    duration = 1
    nn_code = "def model(x):\n    return x\n"

    @classmethod
    def setUpClass(cls):
        prefix = f"unit-sql-{uuid.uuid4().hex[:8]}"
        cls.stats_task = f"{prefix}-stats"
        cls.legacy_task = f"{prefix}-legacy"
        cls.shared_nn = f"{prefix}-shared"
        cls.legacy_nns = (
            f"{prefix}-legacy-a",
            f"{prefix}-legacy-b",
            f"{prefix}-legacy-c",
        )
        cls.prm_ids: list[str] = []

        cls.stats_prm_ids = (
            cls._save_trial(
                task=cls.stats_task,
                nn_name=cls.shared_nn,
                accuracy=0.51,
                total_params=111,
            ),
            cls._save_trial(
                task=cls.stats_task,
                nn_name=cls.shared_nn,
                accuracy=0.73,
                total_params=222,
            ),
        )

        cls._save_trial(task=cls.legacy_task, nn_name=cls.legacy_nns[0], accuracy=0.55)
        cls._save_trial(task=cls.legacy_task, nn_name=cls.legacy_nns[1], accuracy=0.72)
        cls._save_trial(task=cls.legacy_task, nn_name=cls.legacy_nns[2], accuracy=0.91)

    @classmethod
    def tearDownClass(cls):
        conn, cur = DB_Init.sql_conn()
        try:
            cur.execute("DELETE FROM stat WHERE task IN (?, ?)", (cls.stats_task, cls.legacy_task))
            for prm_id in cls.prm_ids:
                cur.execute("DELETE FROM prm WHERE uid = ?", (prm_id,))
                cur.execute("DELETE FROM nn_stat WHERE prm_id = ?", (prm_id,))
            cur.execute(
                "DELETE FROM nn WHERE name IN (?, ?, ?, ?)",
                (cls.shared_nn, *cls.legacy_nns),
            )
            conn.commit()
        finally:
            DB_Init.close_conn(conn)

    @classmethod
    def _save_trial(
        cls,
        *,
        task: str,
        nn_name: str,
        accuracy: float,
        total_params: int | None = None,
    ) -> str:
        prm_id = f"{task}-{nn_name}-{uuid.uuid4().hex[:10]}"
        prm = {
            "uid": prm_id,
            "lr": round(accuracy, 3),
            "batch": 8,
            "transform": cls.transform,
            "epoch": cls.epoch,
            "duration": cls.duration,
            "accuracy": accuracy,
        }
        DB_Write.save_nn(
            nn_code=cls.nn_code,
            task=task,
            dataset=cls.dataset,
            metric=cls.metric,
            epoch=cls.epoch,
            prm=prm,
            force_name=nn_name,
        )
        if total_params is not None:
            DB_Write.save_nn_stat(
                nn_name,
                prm_id,
                {
                    "total_params": total_params,
                    "trainable_params": total_params,
                },
            )
        cls.prm_ids.append(prm_id)
        return prm_id

    def test_include_nn_stats_matches_prm_id(self):
        rows = DB_Read.data(
            task=self.stats_task,
            dataset=self.dataset,
            metric=self.metric,
            nn=self.shared_nn,
            include_nn_stats=True,
        )

        self.assertEqual(2, len(rows))
        stats_by_prm = {row["prm_id"]: row["nn_total_params"] for row in rows}
        self.assertEqual(111, stats_by_prm[self.stats_prm_ids[0]])
        self.assertEqual(222, stats_by_prm[self.stats_prm_ids[1]])

    def test_legacy_pairwise_prefers_best_higher_accuracy_match(self):
        rows = DB_Read.data(
            task=self.legacy_task,
            dataset=self.dataset,
            metric=self.metric,
            max_rows=10,
            sql=JoinConf(
                num_joint_nns=2,
                same_columns=("task", "dataset", "metric", "epoch"),
                diff_columns=("nn",),
                enhance_nn=True,
            ),
        )

        by_nn = {row["nn"]: row for row in rows}
        self.assertEqual(self.legacy_nns[2], by_nn[self.legacy_nns[0]]["nn_2"])
        self.assertEqual(self.legacy_nns[2], by_nn[self.legacy_nns[1]]["nn_2"])
        self.assertIsNone(by_nn[self.legacy_nns[2]]["nn_2"])


    def test_legacy_performance_smoke(self):
        rows = DB_Read.data(
            task=self.legacy_task,
            dataset=self.dataset,
            metric=self.metric,
            sql=JoinConf(
                num_joint_nns=2,
                diff_columns=("nn",),
            ),
            max_rows=10,
        )
        self.assertIsInstance(rows, tuple)
        for row in rows:
            self.assertIn("nn", row)
            self.assertIn("nn_2", row)
            if row["nn_2"] is not None:
                self.assertNotEqual(row["nn"], row["nn_2"])


if __name__ == "__main__":
    unittest.main()
