"""Tests of the validation split (ab.nn.util.Split, load_dataset with the split by default and test_only=True, Train with
a validation set, the --test_only flag).

CPU only, no download (fake_cifar stands in for cifar-10). Run from an empty working directory: nn-dataset puts db/,
out/ and stat files under it."""
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import fake_cifar  # noqa: E402

from ab.nn.util import Split  # noqa: E402


class _DS(torch.utils.data.Dataset):
    """torchvision-like dataset: .targets, .transform, .target_transform; item = (transform(i), target)."""

    def __init__(self, targets, transform):
        self.targets = list(targets)
        self.transform = transform
        self.target_transform = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(i), self.targets[i]


# ---------------------------------------------------------------- sizes
@pytest.mark.parametrize("n_train,n_test,expected", [
    (50000, 10000, 10000),   # cifar-10 / cifar-100
    (60000, 10000, 10000),   # mnist
    (73257, 26032, 14651),   # svhn: cap 0.2 binds
    (162770, 19962, 19962),  # celeba
    (10, 100, 2),
])
def test_val_size(n_train, n_test, expected):
    assert Split.val_size(n_train, n_test) == expected


def test_split_sizes_no_overlap_cover():
    tr, va = Split.split_indices(1000, 200, labels=np.arange(1000) % 7, seed=3)
    assert len(va) == 200 and len(tr) == 800
    assert not set(tr) & set(va)
    assert sorted(set(tr) | set(va)) == list(range(1000))


def test_stratified_proportions():
    labels = np.array([0] * 600 + [1] * 300 + [2] * 100)
    tr, va = Split.split_indices(1000, 100, labels=labels, seed=1)
    counts = np.bincount(labels[va], minlength=3)
    assert counts.tolist() == [60, 30, 10]
    # uneven shares: largest remainder keeps the total exact and each class within one of its share
    labels = np.array([0] * 7 + [1] * 5 + [2] * 3)
    tr, va = Split.split_indices(15, 4, labels=labels, seed=1)
    counts = np.bincount(labels[va], minlength=3)
    assert counts.sum() == 4
    assert all(abs(c - share) < 1 for c, share in zip(counts, [7 * 4 / 15, 5 * 4 / 15, 3 * 4 / 15]))


def test_seeded_reproducible_and_seed_dependent():
    labels = np.arange(500) % 10
    a = Split.split_indices(500, 100, labels=labels, seed=42)
    b = Split.split_indices(500, 100, labels=labels, seed=42)
    c = Split.split_indices(500, 100, labels=labels, seed=43)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[1], c[1])
    # independent of torch's and numpy's global random state
    torch.manual_seed(123)
    np.random.seed(123)
    d = Split.split_indices(500, 100, labels=labels, seed=42)
    assert np.array_equal(a[1], d[1])


def test_unlabelled_split():
    tr, va = Split.split_indices(100, 10, labels=None, seed=0)
    assert len(va) == 10 and not set(tr) & set(va)


def test_bad_sizes():
    with pytest.raises(ValueError):
        Split.split_indices(10, 0)
    with pytest.raises(ValueError):
        Split.split_indices(10, 10)


# ---------------------------------------------------------------- transforms
def test_val_uses_test_transform_train_keeps_train_transform():
    train = _DS(np.arange(100) % 5, transform=lambda i: ("train", i))
    test = _DS(np.arange(20) % 5, transform=lambda i: ("test", i))
    tr, va = Split.split_train_val(train, test, labels=np.asarray(train.targets))
    assert len(va) == 20 and len(tr) == 80
    assert all(va[k][0][0] == "test" for k in range(len(va)))
    assert all(tr[k][0][0] == "train" for k in range(len(tr)))
    # val items come from the training split (indices), not from the test set
    assert {va[k][0][1] for k in range(len(va))} <= set(range(100))
    assert not {tr[k][0][1] for k in range(len(tr))} & {va[k][0][1] for k in range(len(va))}
    # the original datasets are not modified
    assert train.transform(0)[0] == "train" and test.transform(0)[0] == "test" and len(test) == 20


def test_eval_view_needs_transform():
    with pytest.raises(TypeError):
        Split.eval_view(object(), object())


def test_dataset_labels():
    ds = fake_cifar.FakeVision(30, 0)
    assert Split.dataset_labels("cifar-10", ds).tolist() == ds.targets
    assert Split.dataset_labels("imagenette", ds) is None


def test_loaded_dataset_is_a_4_tuple_with_val_set():
    r = Split.LoadedDataset((10,), 0.1, "train", "test", "val")
    out_shape, min_acc, train_set, test_set = r
    assert len(r) == 4 and tuple(r) == ((10,), 0.1, "train", "test") and r.val_set == "val"
    assert Split.LoadedDataset((10,), 0.1, "train", "test").val_set is None


@pytest.mark.parametrize("name", ["imagenette", "wikitext", "utkface", "coco"])
def test_other_datasets_fall_back_without_error(name, capsys):
    train, test = fake_cifar.FakeVision(40, 0), fake_cifar.FakeVision(10, 1)
    r = Split.with_validation(name, ((10,), 0.1, train, test))
    assert tuple(r) == ((10,), 0.1, train, test) and r.val_set is None   # the loader's sets, unchanged
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and name in out[0] and "no true test split" in out[0]


@pytest.mark.parametrize("name", Split.TRUE_TEST_LOADERS[:3])
def test_true_test_datasets_split(name):
    train, test = fake_cifar.FakeVision(400, 0), fake_cifar.FakeVision(100, 1)
    r = Split.with_validation(name, ((10,), 0.1, train, test))
    assert len(r.val_set) == 80 and len(r[2]) == 320 and r[3] is test   # cap 20% of 400 binds


# ---------------------------------------------------------------- load_dataset
@pytest.fixture
def workdir():
    """The working directory, which must be fresh and empty except for what this fixture creates: nn-dataset fixes
    its root (db/, out/) at import time from the working directory. run_val_tests.sh starts one pytest process per
    test that uses it, each in a new empty directory."""
    base = Path.cwd()
    from ab.nn.util.Const import db_file
    assert Path(db_file).parent.parent == base, db_file      # never a real database
    assert not (base / "db").exists(), "workdir tests need a fresh working directory (one test per process)"
    (base / "tr").mkdir()
    (base / "tr" / "ident.py").write_text("def transform(norm):\n    return lambda x: x\n")
    fake_cifar.install()
    fake_cifar.prepare_workdir(base)
    (base / "db").mkdir()
    from ab.nn.util.db.Init import init_db
    init_db()
    return base


def test_load_dataset_test_only_is_the_loader_result(workdir):
    from ab.nn.util.Loader import load_dataset
    res = load_dataset("img-classification", "cifar-10", "ident", str(workdir / "tr"), test_only=True)
    assert type(res) is tuple and len(res) == 4 and getattr(res, "val_set", None) is None
    out_shape, min_acc, train_set, test_set = res
    assert isinstance(train_set, fake_cifar.FakeVision) and len(train_set) == fake_cifar.N_TRAIN
    assert len(test_set) == fake_cifar.N_TEST


def test_load_dataset_default_splits(workdir):
    from ab.nn.util.Loader import load_dataset
    res = load_dataset("img-classification", "cifar-10", "ident", str(workdir / "tr"))
    assert len(res) == 4                       # existing callers unpack four values
    out_shape, min_acc, train_set, test_set = res
    val_set = res.val_set
    n_val = Split.val_size(fake_cifar.N_TRAIN, fake_cifar.N_TEST)
    assert len(val_set) == n_val and len(train_set) == fake_cifar.N_TRAIN - n_val
    assert len(test_set) == fake_cifar.N_TEST and isinstance(test_set, fake_cifar.FakeVision)
    assert not set(train_set.indices) & set(val_set.indices)
    # stratified: 10 balanced classes -> n_val / 10 each
    labels = np.asarray(val_set.dataset.targets)[val_set.indices]
    assert np.bincount(labels).tolist() == [n_val // 10] * 10
    assert train_set.num_workers == 0 and val_set.num_workers == 0


def test_load_dataset_other_loader_falls_back(workdir, capsys):
    from ab.nn.util.Loader import load_dataset
    sys.modules["ab.nn.loader.utkface"] = fake_cifar   # stand-in, no download: a loader without a true test split
    res = load_dataset("img-classification", "utkface", "ident", str(workdir / "tr"))
    assert len(res) == 4 and res.val_set is None
    assert len(res[2]) == fake_cifar.N_TRAIN and len(res[3]) == fake_cifar.N_TEST
    assert "no true test split" in capsys.readouterr().out


def test_optuna_objective_passes_val_and_test_only(workdir, monkeypatch):
    import optuna
    from ab.nn.util import Train as T
    seen = []

    class FakeTrain:
        def __init__(self, *a, **k):
            seen.append((a, k))

        def train_n_eval(self, *a, **k):
            return 0.5, 0.1, 1

    real = T.load_dataset
    monkeypatch.setattr(T, "load_dataset", lambda task, ds, tr, **kw: real(task, ds, tr, str(workdir / "tr"), **kw))
    monkeypatch.setattr(T, "get_ab_nn_attr", lambda *a: (lambda: {"lr", "momentum"}))
    monkeypatch.setattr(T, "nn_mod", lambda *a: "fake")
    monkeypatch.setattr(T, "Train", FakeTrain)
    prm = {"lr": 0.01, "momentum": 0.9, "batch": 32, "transform": "ident"}
    n_val = Split.val_size(fake_cifar.N_TRAIN, fake_cifar.N_TEST)
    for test_only in (False, True):
        T.optuna_objective(optuna.trial.FixedTrial({}), ("img-classification", "cifar-10", "acc", "fake"), prm, 0,
                           1e-4, 1e-1, 0.0, 1.0, 0.0, 0.5, 4, 6, None, 1, 1, None, 30, False, False, test_only=test_only)
    (a0, k0), (a1, k1) = seen
    assert len(k0["val_dataset"]) == n_val and len(a0[6]) == fake_cifar.N_TRAIN - n_val
    assert k1["val_dataset"] is None and len(a1[6]) == fake_cifar.N_TRAIN


def test_optuna_objective_maximises_val_by_default(workdir, monkeypatch):
    """Objective value: validation accuracy with the split (default), test accuracy with test_only=True; also for a
    trial rejected by the minimum-accuracy check."""
    import optuna
    from ab.nn.util import Train as T
    from ab.nn.util.Exception import AccuracyException
    mode = {"raise": False}

    class FakeTrain:
        def __init__(self, *a, **k):
            self.last_val_accuracy = 0.8 if k.get("val_dataset") is not None else None

        def train_n_eval(self, *a, **k):
            if mode["raise"]:
                raise AccuracyException(0.3, 1, "too low")
            return 0.5, 0.1, 1            # test accuracy, as recorded

    real = T.load_dataset
    monkeypatch.setattr(T, "load_dataset", lambda task, ds, tr, **kw: real(task, ds, tr, str(workdir / "tr"), **kw))
    monkeypatch.setattr(T, "get_ab_nn_attr", lambda *a: (lambda: {"lr", "momentum"}))
    monkeypatch.setattr(T, "nn_mod", lambda *a: "fake")
    monkeypatch.setattr(T, "Train", FakeTrain)
    prm = {"lr": 0.01, "momentum": 0.9, "batch": 32, "transform": "ident"}

    def objective(test_only):
        return T.optuna_objective(optuna.trial.FixedTrial({}), ("img-classification", "cifar-10", "acc", "fake"), prm, 0,
                                  1e-4, 1e-1, 0.0, 1.0, 0.0, 0.5, 4, 6, None, 1, 1, None, 30, False, False,
                                  test_only=test_only)[0]
    assert objective(False) == 0.8 and objective(True) == 0.5
    mode["raise"] = True
    assert objective(False) == 0.8 and objective(True) == 0.3


# ---------------------------------------------------------------- Train with a validation set
def _scripted_eval(monkeypatch, test_accs, val_accs):
    """Train.eval returns scripted accuracies: test_accs for the test loader, val_accs for the val loader."""
    from ab.nn.util import Train as T
    calls = {"test": 0, "val": 0}

    def fake_eval(self, loader):
        kind = "val" if loader is self.val_loader else "test"
        seq = val_accs if kind == "val" else test_accs
        v = seq[calls[kind]]
        calls[kind] += 1
        return v, {self.primary_metric: v}
    monkeypatch.setattr(T.Train, "eval", fake_eval)


def _train(workdir, val=True, epochs=3):
    """train_new with the default path (validation split) or test_only=True (val=False)."""
    from ab.nn.util import Train as T
    prm = dict(fake_cifar.PRM, epoch=epochs)
    return T.train_new(fake_cifar.NET_CODE, "img-classification", "cifar-10", "acc", prm, save_to_db=True,
                       prefix="t", save_path=str(workdir / "stat"), transform_dir=str(workdir / "tr"),
                       return_val=True, **({} if val else {"test_only": True}))


def test_train_val_stats_json_db(workdir, monkeypatch):
    _scripted_eval(monkeypatch, test_accs=[0.9, 0.2, 0.5], val_accs=[0.3, 0.8, 0.6])
    name, acc, acc_time, val_acc = _train(workdir)
    assert acc == 0.5 and val_acc == 0.6          # returned accuracy stays test; val is the 4th value
    stats = {int(f.stem): json.loads(f.read_text()) for f in (workdir / "stat").glob("*.json")}
    assert sorted(stats) == [1, 2, 3]
    for ep, t, v in ((1, 0.9, 0.3), (2, 0.2, 0.8), (3, 0.5, 0.6)):
        rec = stats[ep][0] if isinstance(stats[ep], list) else stats[ep]
        assert rec["accuracy"] == t and rec["val_accuracy"] == v
        assert rec["train_stat"]["val_accuracy"] == v and "val_loss" in rec["train_stat"]
    last = stats[3][0] if isinstance(stats[3], list) else stats[3]
    assert last["train_stat"]["best_epoch"] == 2 and last["train_stat"]["best_accuracy"] == 0.8   # best by val
    con = sqlite3.connect(workdir / "db" / "ab.nn.db")
    assert con.execute("select count(*) from prm where name in ('val_accuracy', 'val_loss')").fetchone()[0] == 0
    cols = [r[1] for r in con.execute("PRAGMA table_info(train_stat)")]
    assert "val_accuracy" in cols and "val_loss" in cols
    vals = sorted(r[0] for r in con.execute("select val_accuracy from train_stat"))
    assert vals == [0.3, 0.6, 0.8]
    assert sorted(r[0] for r in con.execute("select accuracy from stat")) == [0.2, 0.5, 0.9]
    summary = json.loads((workdir / "out" / "training_summary.json").read_text())
    assert [e["val_accuracy"] for e in summary["epoch_details"]] == [0.3, 0.8, 0.6]


def test_minimum_accuracy_check_uses_val(workdir, monkeypatch):
    from ab.nn.util.Exception import AccuracyException
    monkeypatch.setattr(fake_cifar, "MINIMUM_ACCURACY", 0.4)
    # test high, val below the minimum: refused
    _scripted_eval(monkeypatch, test_accs=[0.9], val_accs=[0.1])
    with pytest.raises(AccuracyException):
        _train(workdir, epochs=1)


def test_minimum_accuracy_check_passes_on_val(workdir, monkeypatch):
    monkeypatch.setattr(fake_cifar, "MINIMUM_ACCURACY", 0.4)
    # test below the minimum, val above it: accepted (the decision reads val only)
    _scripted_eval(monkeypatch, test_accs=[0.1], val_accs=[0.9])
    name, acc, acc_time, val_acc = _train(workdir, epochs=1)
    assert acc == 0.1 and val_acc == 0.9


def test_no_val_keeps_old_records(workdir, monkeypatch):
    _scripted_eval(monkeypatch, test_accs=[0.9, 0.2], val_accs=[])
    res = _train(workdir, val=False, epochs=2)
    assert len(res) == 4 and res[3] is None   # return_val=True without a split: None
    for f in (workdir / "stat").glob("*.json"):
        rec = json.loads(f.read_text())
        rec = rec[0] if isinstance(rec, list) else rec
        assert "val_accuracy" not in rec and "val_accuracy" not in rec["train_stat"]
        assert rec["train_stat"]["best_epoch"] == 1
    con = sqlite3.connect(workdir / "db" / "ab.nn.db")
    cols = [r[1] for r in con.execute("PRAGMA table_info(train_stat)")]
    assert "val_accuracy" not in cols


def test_check_nn_default_signature_unchanged(workdir, monkeypatch):
    import ab.nn.api as api
    _scripted_eval(monkeypatch, test_accs=[0.7, 0.4], val_accs=[0.6])
    res = api.check_nn(fake_cifar.NET_CODE, "img-classification", "cifar-10", "acc", dict(fake_cifar.PRM, epoch=1),
                       save_to_db=False, prefix="t", transform_dir=str(workdir / "tr"))
    assert len(res) == 3 and res[1] == 0.7                  # default (validation split): three values, test accuracy
    res = api.check_nn(fake_cifar.NET_CODE, "img-classification", "cifar-10", "acc", dict(fake_cifar.PRM, epoch=1),
                       save_to_db=False, prefix="t", transform_dir=str(workdir / "tr"), test_only=True)
    assert len(res) == 3 and res[1] == 0.4


# ---------------------------------------------------------------- --test_only flag (imports ab.nn.train: needs the workdir DB)
def test_cli_test_only_flag(workdir, monkeypatch):
    from ab.nn.util import Util
    monkeypatch.setattr(sys, "argv", ["train"])
    assert Util.args().test_only is False
    monkeypatch.setattr(sys, "argv", ["train", "--test_only"])
    assert Util.args().test_only is True


def test_train_main_passes_test_only(workdir):
    import inspect
    import ab.nn.train as tr
    assert inspect.signature(tr.main).parameters["test_only"].default is False
    src = inspect.getsource(tr)
    assert "test_only=test_only" in src and "a.test_only" in src
