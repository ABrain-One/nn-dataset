# Architecture identity (`arch_uid`) — reference

Everything about the architecture fingerprint, the duplicate-model merge that was applied
with it, the tests behind both, and how to run any of it again. Paths are relative to the
repository root. All commands assume the repository root as the working directory and the
usual environment (`pip install -r requirements.txt`).

## 1. Why this exists

nn-dataset identifies a model by `uuid4(code)` in `ab/nn/util/Util.py` — the md5 of the
source text with whitespace removed. That is a *content* hash: the same network written
twice with different variable names, comments or formatting gets two identities, two files
and two sets of statistics that nothing in the schema connects. `arch_uid` is a second
identity that answers a different question — *is this the same architecture?* — and sits
beside the md5, which stays the primary key.

The merge described in §5 removed 2,701 model files that were the same architecture as
another file, moving every measurement onto the surviving name.

## 2. Files

### `ab/nn/util/ArchUID.py` — the fingerprint
Turns a model's source text into a certificate of its architecture.

How: `ast.parse` the file (never imported or executed); build one def-use graph over every
function in it, where variables are edges rather than nodes (so names vanish), literal
values are folded into node parameters, attribute reads are wired to their producers per
class, module-level constants and definitions are visible inside methods, calls to
file-local classes/functions link to a definition node, and containers built by
`append`/loops/comprehensions are expanded so they match the written-out form. The graph
is then hashed by Weisfeiler–Lehman refinement (4 rounds, `blake2b`), giving a
32-character hex `arch_uid`. Everything the certificate hashes is exact (Python ints,
floats, strings); numpy is used only for the optional graded measures.

| entry point | returns |
|---|---|
| `arch_uid(src: str) -> str` | the certificate; equal iff the architectures are the same |
| `arch_similarity(a, b) -> float` | Jaccard over WL subtree labels, 0..1, for reporting/ranking only — never for identity |
| `arch_vec(src) -> np.ndarray` | unit-norm continuous embedding (kept for research use) |
| `source_graph(src)` / `wl_hash(g)` | the two halves, for inspection |

Deliberately not attempted: dead-code elimination, sorting of unordered constructs,
inter-procedural value flow beyond default arguments.

### `ab/nn/util/ArchDedup.py` — the duplicate gate for generation loops
Answers "has this architecture been seen?" for a candidate model file in one dictionary
lookup — no pairwise comparison, no instantiation, ~5–8 ms.

    python -m ab.nn.util.ArchDedup --build            # once per DB: index {arch_uid: name} -> out/arch_index.json
    python -m ab.nn.util.ArchDedup --check FILE       # dataset | run | new | unparseable, with the colliding name
    python -m ab.nn.util.ArchDedup --explain FILE     # print the graph the identity is computed from
    python -m ab.nn.util.ArchDedup --sql              # print the additive schema change (§7)

In code, placed before the budget gate of a generation loop:

    from ab.nn.util.ArchDedup import ArchIndex
    idx = ArchIndex.load()                 # once per run
    v = idx.check(src)                     # per candidate
    if v.duplicate: reject(v.reason)       # "already in dataset as X" / "already this run as Y"
    else: idx.accept(v, name)              # so later candidates in this run see it

Index and DB paths come from `ab.nn.util.Const` (`db_file`, `out_dir`).

### `ab/nn/util/ArchUIDMerge.py` — the one-off merge
Collapses duplicate architectures in the file tree while keeping every measurement.

    python -m ab.nn.util.ArchUIDMerge                     # PLAN: report only, nothing written
    python -m ab.nn.util.ArchUIDMerge --show 3            # ... and detail the 3 largest groups
    python -m ab.nn.util.ArchUIDMerge --apply --log merge_log.json
    python -m ab.nn.util.ArchUIDMerge --root /other/checkout/ab/nn

What it does, in order:
1. fingerprints every `ab/nn/nn/*.py`; files that do not parse are listed and never merged;
2. groups by `arch_uid`; in each group the **survivor** is the member with the most
   recorded training runs (name as tie-break), the others are **removed**;
3. **preflight**: refuses to write anything if any group member is missing on disk;
4. `stat/train/<task>_<dataset>_<metric>_<model>/<epoch>.json` — directories of a removed
   model are renamed onto the survivor, or merged into the survivor's existing directory;
   runs are deduplicated on the database's own `stat.id` key (transform, prm uid, duration,
   accuracy), so only byte-identical repeats are dropped;
5. `stat/run/<backend>/<precision>/<config>/…` — same rename/merge; **one timing per device**:
   if the survivor already has a file for that device the incoming one is discarded;
   the `model_name` field *inside* each timing file is rewritten (the `run.id` hash reads it);
6. `stat/run/**/all_models.json` — the survivor's record is kept; a removed model's record
   is moved only if the survivor has none;
7. `stat/nn/<model>.json` — renamed onto the survivor, or removed if the survivor has one;
8. removes the duplicate `.py` files;
9. **conservation**: counts runs on disk before and after and prints the unexplained loss;
10. writes the JSON log (§6).

Running it again on a merged tree finds 0 groups (idempotent).

### `test_arch_uid.py` — the correctness contract (41 identity cases + 6 graded)
Hand-written pairs of model sources with the expected verdict. Run `python test_arch_uid.py`
(prints every case) or `pytest test_arch_uid.py`.

Must hash **SAME** (21):
- attribute rename (self.reduce -> self.proj)
- local temporary rename (y = ... -> z = ...)
- function PARAMETER rename (forward(self, x) -> forward(self, inp))
- whitespace / trailing-paren restyle
- import alias (nn.Conv2d -> torch.nn.Conv2d)
- comment added
- super(Cls, self).__init__() vs super().__init__()
- type annotations added
- docstring added / reworded
- constant folding (Linear(9216,..) vs Linear(256*6*6,..))
- loop-built vs unrolled Sequential (3 identical convs)
- comprehension-built vs unrolled Sequential
- loop variable used in widths vs written-out widths
- DEAD default argument  factor=4 vs 8, every call site passes it
- unused module-level constant added
- subscript store == written-out list  [32,64]; ws[0]=16 vs [16,64]
- helper CLASS renamed consistently  Helper -> Widget
- module-level helper FUNCTION renamed consistently
- local callee: keyword vs positional  Block(64, factor=4) vs Block(64, 4)
- local callee: parameter renamed at def AND call  factor -> f
- self.method(x, probe=True): keyword param renamed at def AND call

Must hash **DIFF** (20):
- channel width 128 -> 256
- kernel size 1 -> 3
- op swapped (ReLU -> GELU)
- layer removed (no BatchNorm)
- backbone choice inside a CHAINED call  .to(device)
- backbone choice
- optimizer SGD -> AdamW  (train_setup)
- grad-clip before vs after step  (learn)
- loop trip count 3 -> 7
- activations SWAPPED between two equal-width blocks
- activations swapped across a SYMMETRIC skip (a + b)
- activations swapped between two SEPARATE classes (same attr name)
- layer ORDER swapped inside the stack
- LIVE default argument  factor=4 vs 8, never passed at the call site
- module-level constant read in a method  MOM 0.1 vs 0.001
- subscript store on a list  ws[0] = 16 vs absent
- hyperparameter override in code  prm['lr'] = 0.001 vs absent
- different helper class used  Helper (conv) vs Other (linear)
- module-level helper function body changed  ReLU -> LeakyReLU
- library call keyword still matters  Conv2d(kernel_size=3) vs (stride=3)

Graded `arch_similarity` ranges (6):
- identical code == 1.0: 1.00–1.00
- pure rename == 1.0: 1.00–1.00
- one layer removed stays high: 0.55–0.99
- chain vs residual: related, not same: 0.45–0.95
- width change is a real drop: 0.05–0.75
- unrelated network is low: 0.00–0.35

### `fuzz_arch_uid.py` — the randomized test
Takes real models from `ab/nn/nn` (or `--dir`), applies each mutation, and checks the
verdict. `python fuzz_arch_uid.py --n 400 --seed 0 --show 3`.

| mutation | must the hash change? |
|---|---|
| rename attributes (`self.x` → `self.zzN`) | no |
| rename locals and parameters (definition and call-site keywords together) | no |
| add docstrings | no |
| add type annotations | no |
| swap two constructor calls | yes |
| change a numeric literal | yes |
| delete a layer assignment | yes |

Expected total failure rate ≈ 0.5–1 %; the residue is fuzzer artefacts (a renamed name
reached through a string, a changed literal that was a dead default).

### `doc/arch_uid/merge_log_2026-09-08.json` — the record of the applied merge
See §6.

## 3. What was verified before the merge was applied

| check | how | result |
|---|---|---|
| identity contract | `test_arch_uid.py` | 47/47 |
| value-flow battery | 14 minimal programs, each value reaching a layer a different way | 0/14 blind spots |
| randomized | `fuzz_arch_uid.py --n 400` (2,794 mutations) | 0.5 % |
| reproducible across environments | same 300 models under numpy 1.21.5 and 2.2.6, and under different `PYTHONHASHSEED` | 0/300 differ |
| independent signal | recorded `total_params` in `stat/nn` across each merged group; every disagreement traced to the dataset the file was instantiated for (class count / input size), none to the architecture | ok |
| remaining default-argument disagreements inside merged groups | all 47 are dead defaults (every call site passes the argument) | 0 live |
| conservation | runs on disk before/after, unexplained loss | 0 |
| database consistency | rebuilt from the merged tree; every `stat`, `run`, `tflite`, `prun`, `nn_stat`, `nn_minhash` row resolves to an existing model | 0 new orphans (4 pre-existing at HEAD) |
| idempotency | tool re-run on the merged tree | 0 groups |
| scope | `git status` | all data paths under `ab/nn/nn` and `ab/nn/stat` |
| reproducibility of the plan | three independent fresh-clone runs, and the tool run from nn-gpt vs from this package | identical `renames`, groups, survivors |
| gate | index built from the merged DB; verbatim model → `dataset`, helper class renamed → `dataset`, real change → `new`, resubmitted → `run` | pass |

## 4. Numbers (origin/main 51c8d261844, 2026-09-08)

| | before | after |
|---|---:|---:|
| model files | 26,077 | **23,376** (−2,701, 10.4 %) |
| duplicate architecture groups | | **1,611** |
| training runs on disk | 1,010,099 | 1,010,098 (one byte-identical repeat) |
| `nn` rows (DB rebuilt) | 25,930 | 23,231 |
| `stat` / `run` / `tflite` / `prun` rows | 1,010,099 / 14,376 / 8,840 / 589 | 1,010,098 / 13,355 / 8,213 / 547 |
| `nn_stat` / `nn_minhash` rows | 16,790 / 13,761 | 15,436 / 12,568 |
| orphan rows | 4 (pre-existing) | 4 (same) |

Per folder: `stat/train` 490 dirs renamed, 4,272 merged, 58,667 runs relocated;
`stat/run` 488 renamed, 3,166 merged, 103 timings relocated, 4,705 discarded;
`all_models.json` 51 keys moved, 126 lost; `stat/nn` 131 renamed, 1,354 removed;
828 embedded `model_name` fields rewritten. Git: 48,383 paths in the data commit.

## 5. Decisions taken (with Dr Ignatov, August 2026)
- `all_models.json` keeps the survivor's record even when a removed duplicate had higher
  accuracy — it keeps the GitHub data consistent with the Hugging Face checkpoints, which
  were trained with the corresponding transform. No format change.
- device timings: one per device per model, the survivor's; no suffixed copies.
- `stat/nn` is treated as intrinsic to the network; duplicates' files are removed.
- the 4 pre-existing orphan rows are left as they were (`SwinIR`, `tmp_40de6cd3`: metric
  files with no model file; `rl-bb-init-65d770c6…`: a model with no training runs).

## 6. The merge log
`ArchUIDMerge --log` writes one JSON document per run:

    root, generated, mode ("plan" | "apply")
    summary      every counter above, plus runs_on_disk_before / _after
    skipped      models the parser could not read
    renames      { removed name: survivor }                 <- the map external systems need
    groups[]     arch_uid, survivor, survivor_runs,
                 removed[]: name, runs, actions[]:
                    stat/train      config, renamed | merged (+runs_relocated, runs_dropped)
                    stat/run/<tree> config, renamed | merged (+files_relocated, files_discarded)
                    all_models      file, moved | lost | identical, the record itself
                    stat/nn         renamed | removed

## 7. After the merge is on `main`

**Rebuild and re-upload the Hugging Face copy of the database.** `init_population()` in
`ab/nn/util/db/Write.py` creates the database in one of two ways: if `ab/nn/stat/` exists
(a git checkout) it builds it from the JSON files; if not (a `pip install nn-dataset`, which
ships no statistics) it calls `db_from_hf()` and downloads a prebuilt `ab.nn.db.zst` from
the Hugging Face repo `NN-Dataset/LEMUR_DB`. That file was built before the merge, so
every pip user keeps seeing the 2,701 removed models until it is regenerated:

    python ab/nn/util/hf/DB2HF.py --HF_TOKEN <token>      # init_population + compress + upload

Also keyed by old names and untouched by the merge: checkpoints on Hugging Face, and the
`_work/` bookkeeping of `ab/nn/imp/upload2HF.py` / `prHF.py`. `renames` in the log maps
them. Open follow-ups: whether to drop the 51 *moved* `all_models.json` keys (they attach
a quantized accuracy to a name whose HF checkpoint is a different file), and the 4 orphans.

**Optional, additive schema change** so duplicates are a query rather than a migration
(`python -m ab.nn.util.ArchDedup --sql`):

    ALTER TABLE nn ADD COLUMN arch_uid TEXT;
    CREATE INDEX IF NOT EXISTS idx_nn_arch_uid ON nn(arch_uid);
    -- compute in Write.py::save_nn only (json_train_to_db discards code_to_db's return)
    SELECT arch_uid, COUNT(*) c, GROUP_CONCAT(name) FROM nn GROUP BY arch_uid HAVING c > 1;

## 8. Re-running everything (fresh clone, ~25 min, mostly two DB builds)

    git clone https://github.com/ABrain-One/nn-dataset X && cd X
    python test_arch_uid.py                                   # expect: all cases hold
    python fuzz_arch_uid.py --n 400 --seed 0                  # expect: total ~0.5 %
    python -m ab.nn.util.ArchUIDMerge --log plan.json         # plan; read it
    python -m ab.nn.util.ArchUIDMerge --apply --log merge_log.json
    python -m ab.nn.util.ArchUIDMerge --log idem.json         # expect: 0 groups
    rm -f db/ab.nn.db && python -c "from ab.nn.util.db.Write import init_population; init_population()"
    python -m ab.nn.util.ArchDedup --build                    # expect: N models -> N architectures, 0 collapsed
    python -m ab.nn.util.ArchDedup --check ab/nn/nn/AirNext.py   # expect: dataset

Orphan audit after a rebuild (SQLite, any client):

    SELECT COUNT(*) FROM stat    WHERE nn         NOT IN (SELECT name FROM nn);
    SELECT COUNT(*) FROM run     WHERE model_name NOT IN (SELECT name FROM nn);
    SELECT COUNT(*) FROM tflite  WHERE model_name NOT IN (SELECT name FROM nn);
    SELECT COUNT(*) FROM prun    WHERE model_name NOT IN (SELECT name FROM nn);
    SELECT COUNT(*) FROM nn_stat WHERE nn_name    NOT IN (SELECT name FROM nn);

If `main` has moved since the last merge, the plan simply includes the new models.
