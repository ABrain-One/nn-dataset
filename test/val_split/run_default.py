"""Tiny CPU training through train_new (save_to_db True), from an empty working directory (nn-dataset puts db/, out/
and stat files under the working directory). Prints the outputs as JSON with the timing and resource fields removed,
so the original and the patched nn-dataset can be compared.

usage: python run_default.py WORKDIR [--test_only] [--val]   (WORKDIR must be empty or missing)
  no flag: train_new's default arguments (unmodified upstream: train/test; patched: validation split)
  --test_only: test_only=True (patched only; must equal unmodified upstream with no flag)
  --val: also return_val=True (patched only, record of the validation path)"""
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

work = Path(sys.argv[1]).absolute()
use_val = "--val" in sys.argv
use_test_only = "--test_only" in sys.argv
work.mkdir(parents=True, exist_ok=True)
assert not any(work.iterdir()), f"{work} is not empty"
os.chdir(work)
(work / "tr").mkdir()
(work / "tr" / "ident.py").write_text("def transform(norm):\n    return lambda x: x\n")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import fake_cifar

fake_cifar.install()
fake_cifar.prepare_workdir(work)
(work / "db").mkdir()
from ab.nn.util.db.Init import init_db
init_db()   # empty DB in WORKDIR, so nothing is imported from the package's stat folder
from ab.nn.util import Train

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
kw = dict(return_val=True) if use_val else {}
if use_test_only:
    kw["test_only"] = True
res = Train.train_new(fake_cifar.NET_CODE, "img-classification", "cifar-10", "acc", dict(fake_cifar.PRM),
                      save_to_db=True, prefix="t", save_path=str(work / "stat"), transform_dir=str(work / "tr"), **kw)

VOLATILE = {"duration", "duration_seconds", "samples_per_second", "cpu_usage_percent", "occupied_ram_kb",
            "ram_usage_percent", "gpu_memory_kb", "occupied_gpu_memory_kb", "gpu_memory_usage_percent",
            "total_ram_kb", "cpu_count", "cpu_type", "gpu_type", "gpu_total_memory_kb", "final_resource_usage",
            "system_info", "accuracy_to_time", "id", "stat_id"}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in sorted(o.items()) if k not in VOLATILE}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float):
        return round(o, 6)
    return o


out = {"return": [res[0], round(res[1], 6)] + ([res[3]] if use_val else []),
       "return_len": len(res), "stat_files": {}, "db": {}}
for f in sorted((work / "stat").glob("*.json")):
    out["stat_files"][f.name] = clean(json.loads(f.read_text()))
summary = work / "out" / "training_summary.json"
if summary.exists():
    out["training_summary"] = clean(json.loads(summary.read_text()))
con = sqlite3.connect(work / "db" / "ab.nn.db")
for table in ("stat", "prm", "train_stat"):
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    rows = [dict(zip(cols, r)) for r in con.execute(f"select * from {table}")]
    if table == "prm":
        rows = [r for r in rows if r.get("name") not in VOLATILE]   # timing stored as prm rows (pre-existing)
    out["db"][table] = {"columns": cols, "rows": sorted((json.dumps(clean(r), sort_keys=True, default=str)
                                                         for r in rows))}
print(json.dumps(out, indent=1, sort_keys=True, default=str))
