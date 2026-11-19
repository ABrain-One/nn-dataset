"""The real CLI path: python -m ab.nn.train as a user runs it, in a fresh project directory (fresh db/, no stat
records), with a tiny cifar-10 stand-in (fake_cifar, CPU, no download) and a tiny model. Checks the per-epoch stat
JSON (accuracy_val at top level directly after accuracy, not in train_stat; loss_val in train_stat), the record count
(one record per epoch per run), the database row, and --test_only (no accuracy_val anywhere).

The project is a copy of the ab.nn code (without the model zoo and the statistics) next to a README.md, the files
nn-dataset uses to find its root; cifar-10 is replaced by fake_cifar inside that copy only."""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_cifar  # noqa: E402

CONF = "img-classification_cifar-10_acc_TinyNet"
PRM = {"lr": 0.05, "momentum": 0.9, "batch": 32}


def _project(base: Path) -> Path:
    """A fresh nn-dataset project: ab.nn code, the tiny model, an identity transform, the cifar-10 stand-in, an
    empty stat/train directory (so the fresh database is built locally, not downloaded)."""
    import ab.nn
    src = Path(ab.nn.__file__).parent
    proj = base / "proj"
    nn = proj / "ab" / "nn"
    nn.mkdir(parents=True)
    (proj / "README.md").write_text("test project\n")
    shutil.copy(src.parent / "__init__.py", proj / "ab" / "__init__.py")
    for f in ("__init__.py", "train.py", "api.py"):
        shutil.copy(src / f, nn / f)
    for d in ("util", "loader", "metric"):
        shutil.copytree(src / d, nn / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(Path(fake_cifar.__file__), nn / "loader" / "cifar-10.py")
    for d in ("nn", "transform"):
        (nn / d).mkdir()
        shutil.copy(src / d / "__init__.py", nn / d / "__init__.py")
    (nn / "nn" / "TinyNet.py").write_text(fake_cifar.NET_CODE)
    (nn / "transform" / "ident.py").write_text("def transform(norm):\n    return lambda x: x\n")
    (nn / "stat" / "train").mkdir(parents=True)
    return proj


def _run(proj: Path, *extra):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [sys.executable, "-m", "ab.nn.train", "-c", CONF, "-f", "ident", "-p", json.dumps(PRM), *extra]
    r = subprocess.run(cmd, cwd=proj, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert f"LEMUR root {proj}" in r.stdout, r.stdout[:2000]          # the copy, not an installed nn-dataset
    return r.stdout + r.stderr


def _records(proj: Path):
    files = sorted((proj / "ab" / "nn" / "stat" / "train" / CONF).glob("*.json"))
    assert [f.name for f in files] == ["1.json"]
    recs = json.loads(files[0].read_text())
    return recs if isinstance(recs, list) else [recs]


def _db(proj: Path):
    return sqlite3.connect(proj / "db" / "ab.nn.db")


def test_cli_default_writes_accuracy_val_next_to_accuracy(tmp_path):
    proj = _project(tmp_path)
    out = _run(proj)
    assert "[val-split]" not in out                                    # cifar-10 has a true test split
    recs = _records(proj)
    assert len(recs) == 1                                              # one run, one epoch: one record
    rec = recs[0]
    keys = list(rec)
    assert keys.count("accuracy_val") == 1 and keys.index("accuracy_val") == keys.index("accuracy") + 1
    assert isinstance(rec["accuracy_val"], float)
    assert "accuracy_val" not in rec["train_stat"] and "loss_val" in rec["train_stat"]
    con = _db(proj)
    rows = con.execute("SELECT s.id, s.accuracy, t.accuracy_val, t.loss_val FROM stat s JOIN train_stat t "
                       "ON t.stat_id = s.id WHERE s.nn = 'TinyNet'").fetchall()
    assert len(rows) == 1
    assert rows[0][1] == rec["accuracy"] and rows[0][2] == rec["accuracy_val"]
    assert rows[0][3] == rec["train_stat"]["loss_val"]
    assert con.execute("SELECT COUNT(*) FROM prm WHERE name IN ('accuracy_val', 'loss_val')").fetchone()[0] == 0

    _run(proj)                                                         # a second run adds one record, keeps the first
    recs2 = _records(proj)
    assert len(recs2) == 2 and rec["uid"] in {r["uid"] for r in recs2}
    assert _db(proj).execute("SELECT COUNT(*) FROM stat WHERE nn = 'TinyNet'").fetchone()[0] == 2


def test_cli_test_only_has_no_accuracy_val(tmp_path):
    proj = _project(tmp_path)
    _run(proj, "--test_only")
    recs = _records(proj)
    assert len(recs) == 1
    rec = recs[0]
    assert "accuracy_val" not in rec
    assert "accuracy_val" not in rec["train_stat"] and "loss_val" not in rec["train_stat"]
    cols = [r[1] for r in _db(proj).execute("PRAGMA table_info(train_stat)")]
    assert "accuracy_val" not in cols and "loss_val" not in cols


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
