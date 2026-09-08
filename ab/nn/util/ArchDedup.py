"""Architecture-level duplicate rejection, cheap enough to run before training.

The point of an exact certificate rather than a similarity score is cost. A
candidate is identified by a single hash, so deciding whether it already exists
is one dictionary lookup -- the same cost against twelve thousand models as
against twelve million. Nothing is compared pairwise, and no model is
instantiated, imported or trained to make the decision.

Where it belongs in the loop::

    generate -> parse/assemble -> [ARCH DEDUP] -> budget gate -> train -> eval -> DB

It runs before the budget gate because it is the cheaper of the two (pure text
parsing; the budget gate has to build the module to count parameters), and
before training because a duplicate that reaches training costs a full
evaluation and returns no information.

Three questions are answered by the same lookup:

* has the *dataset* seen this architecture before?
* has this *run* already produced it?
* did this *cycle* produce it twice?

Usage::

    # one-off: build the index from the dataset (minutes, then cached)
    python -m ab.nn.util.ArchDedup --build

    # check a generated file
    python -m ab.nn.util.ArchDedup --check path/to/new_nn.py

    # in the pipeline
    from ab.nn.util.ArchDedup import ArchIndex
    idx = ArchIndex.load()
    v = idx.check(src)
    if v.duplicate:
        reject(v.reason)
    else:
        idx.accept(v, name)        # so the rest of the cycle sees it
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ab.nn.util.ArchUID import arch_uid, authored_region, has_scaffold
from ab.nn.util.Const import db_file, out_dir

DB = db_file
INDEX = out_dir / "arch_index.json"


@dataclass
class Verdict:
    """Outcome of one duplicate check."""
    uid: str | None
    status: str                    # 'new' | 'dataset' | 'run' | 'unparseable'
    match: str | None = None       # name of the model it collides with
    head_uid: str | None = None    # authored-region identity, for reporting

    @property
    def duplicate(self) -> bool:
        return self.status in ("dataset", "run")

    @property
    def reason(self) -> str:
        return {
            "new": "novel architecture",
            "dataset": f"architecture already in dataset as {self.match}",
            "run": f"architecture already generated this run as {self.match}",
            "unparseable": "code does not parse",
        }[self.status]


@dataclass
class ArchIndex:
    """Hash set of known architecture certificates.

    `dataset` holds what the DB knew at build time and is treated as read-only;
    `run` accumulates what the current run has accepted. Keeping them apart lets
    a rejection say *where* the collision came from, which matters when
    diagnosing whether a model is regurgitating the dataset or itself.
    """

    dataset: dict = field(default_factory=dict)   # uid -> model name
    run: dict = field(default_factory=dict)       # uid -> model name
    built_at: float = 0.0

    # ---------------------------------------------------------------- build
    @staticmethod
    def build(db_path: Path = DB, out: Path = INDEX, verbose=True) -> "ArchIndex":
        con = sqlite3.connect(db_path)
        rows = con.execute(
            "SELECT name, code FROM nn WHERE code IS NOT NULL AND length(code) > 50"
        ).fetchall()
        con.close()
        idx, bad = {}, 0
        for i, (name, code) in enumerate(rows):
            if verbose and i and i % 2000 == 0:
                print(f"  {i}/{len(rows)} ...", flush=True)
            try:
                idx.setdefault(arch_uid(code), name)
            except Exception:
                bad += 1
        obj = ArchIndex(dataset=idx, built_at=time.time())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"built_at": obj.built_at, "dataset": idx}))
        if verbose:
            print(f"indexed {len(rows) - bad} models -> {len(idx)} distinct "
                  f"architectures ({len(rows) - bad - len(idx)} collapsed), "
                  f"{bad} unparseable")
            print(f"written to {out}")
        return obj

    @staticmethod
    def load(path: Path = INDEX) -> "ArchIndex":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run `python -m ab.nn.util.ArchDedup --build` first")
        d = json.loads(path.read_text())
        return ArchIndex(dataset=d["dataset"], built_at=d.get("built_at", 0.0))

    # ---------------------------------------------------------------- check
    def check(self, src: str) -> Verdict:
        """One lookup. No pairwise comparison, no model instantiation."""
        try:
            uid = arch_uid(src)
        except Exception:
            return Verdict(None, "unparseable")
        head = None
        if has_scaffold(src):
            try:
                head = arch_uid(authored_region(src))
            except Exception:
                head = None
        if uid in self.run:
            return Verdict(uid, "run", self.run[uid], head)
        if uid in self.dataset:
            return Verdict(uid, "dataset", self.dataset[uid], head)
        return Verdict(uid, "new", None, head)

    def accept(self, v: Verdict, name: str) -> None:
        """Record an accepted model so later candidates in the same run see it."""
        if v.uid:
            self.run.setdefault(v.uid, name)

    def __len__(self) -> int:
        return len(self.dataset) + len(self.run)


# --------------------------------------------------------------------------
# Upstream schema change, for when the dataset itself should carry the column.
# Kept as text on purpose: this touches a shared database and is not run here.
# --------------------------------------------------------------------------
UPSTREAM_SQL = """
-- Additive only: the primary key (md5 of code) is untouched, so no model is
-- renamed and no existing row or foreign key changes.
ALTER TABLE nn ADD COLUMN arch_uid TEXT;
CREATE INDEX IF NOT EXISTS idx_nn_arch_uid ON nn(arch_uid);

-- Backfill: UPDATE nn SET arch_uid = ? WHERE name = ?   (one per row)
-- Then deduplication is a query rather than a migration:
--   SELECT arch_uid, COUNT(*) c, GROUP_CONCAT(name)
--   FROM nn GROUP BY arch_uid HAVING c > 1;
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build the index from the DB")
    ap.add_argument("--check", metavar="FILE", help="check one generated model file")
    ap.add_argument("--sql", action="store_true", help="print the upstream schema change")
    ap.add_argument("--explain", metavar="FILE",
                    help="show the graph the identity is computed from")
    args = ap.parse_args()

    if args.explain:
        from ab.nn.util.ArchUID import source_graph, wl_hash
        src = Path(args.explain).read_text(encoding="utf-8", errors="replace")
        g = source_graph(src)
        print(f"{len(g.keys)} nodes, {len(g.edges)} edges\n")
        print(f"{'#':>4}  {'operation':<28}numeric parameters")
        for i, k in enumerate(g.keys):
            hp = ", ".join(f"{s}" for s in sorted(g.hps[i])) or "-"
            print(f"{i:>4}  {k:<28}{hp}")
        print(f"\nedges (producer -> consumer, argument position):")
        for s, d, p, w in sorted(g.edges)[:40]:
            print(f"   {s:>3} -> {d:<3}  port={p}  weight={w}")
        print(f"\narch_uid = {wl_hash(g)}")
        print("\nIdentifiers appear nowhere above: variable and attribute names are"
              "\nedges, not nodes. Two files differing only in naming produce this"
              "\nsame table, and therefore the same identity.")
        return 0

    if args.sql:
        print(UPSTREAM_SQL)
        return 0
    if args.build:
        ArchIndex.build()
        return 0
    if args.check:
        idx = ArchIndex.load()
        src = Path(args.check).read_text(encoding="utf-8", errors="replace")
        t0 = time.perf_counter()
        v = idx.check(src)
        dt = (time.perf_counter() - t0) * 1000
        print(f"uid      : {v.uid}")
        print(f"head uid : {v.head_uid}")
        print(f"status   : {v.status}  ({v.reason})")
        print(f"checked against {len(idx.dataset)} known architectures in {dt:.1f} ms")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
