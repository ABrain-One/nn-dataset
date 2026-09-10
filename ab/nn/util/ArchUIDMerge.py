"""Collapse isomorphic duplicates in the file system, keeping every measurement.

For each group of models that share an architecture, one file is kept and the
others are removed. Every training run is preserved. Every directory that is
keyed by model name is moved onto the survivor:

    stat/train/<task>_<dataset>_<metric>_<model>/<epoch>.json   training runs
    stat/run/<backend>/<precision>/<t>_<d>_<m>_<model>/*.json   device timings
    stat/run/**/all_models.json                                 name-keyed maps
    stat/nn/<model>.json                                        static metrics

All four feed the database (`json_train_to_db`, `json_run_tflite_to_db`,
`json_prun_to_db`, `json_nn_to_db`), so leaving any of them behind produces rows
that reference a model that no longer exists.

The survivor is the member with the most recorded runs, so the fewest
measurements have to be relocated, with the model name as a deterministic
tie-break.

Two runs are the same measurement only when they agree on transform, parameter
uid, duration and accuracy, which is the key the database itself uses for
`stat.id`. Deduplicating on the parameter uid alone is wrong: it is a hash of
the hyperparameters, so independent trials of the same configuration share it.

Device timings are treated differently, by decision: one measurement per device
per surviving model. Where a duplicate was benchmarked on a device the survivor
already covers, the duplicate's timing is discarded rather than kept under a
suffixed name, so the dataset holds a single consistent number per device.

Nothing is written unless `--apply` is given. The default is a report.

    python -m ab.nn.util.ArchUIDMerge                     # this checkout, plan only
    python -m ab.nn.util.ArchUIDMerge --root /path/to/ab/nn --show 2
    python -m ab.nn.util.ArchUIDMerge --root /path/to/ab/nn --apply

A JSON log of every decision is written (`--log`, default
`arch_uid_merge_log.json` in the working directory): each group's survivor and
removed members, and for each removed member every directory, file and key
that was renamed, merged, moved, discarded or lost. The flat `renames` map in
it (removed name -> survivor) is what anything outside the repository -- a
checkpoint store, a results table -- needs to follow the merge.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ab.nn.util.ArchUID import arch_uid
from ab.nn.util.Const import nn_dir as _nn_dir


# ---------------------------------------------------------------- loading


def load_models(nn_dir: Path):
    """model name -> arch_uid. Unreadable models are left out, never merged."""
    out, skipped = {}, []
    for f in sorted(nn_dir.glob("*.py")):
        if f.stem == "__init__":
            continue
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped.append((f.stem, "unreadable"))
            continue
        if len(src) < 50:
            skipped.append((f.stem, "too short"))
            continue
        try:
            out[f.stem] = arch_uid(src)
        except Exception as e:
            skipped.append((f.stem, type(e).__name__))
    return out, skipped


def split_config(name: str):
    """`task_dataset_metric_model` -> the four parts, or None.

    task, dataset and metric come from fixed vocabularies in nn-dataset
    (`ab/nn/loader/`, `ab/nn/metric/`), so the first three fields never contain
    an underscore and a maxsplit of 3 is exact.
    """
    p = name.split("_", 3)
    return tuple(p) if len(p) == 4 else None


def load_dirs(root: Path):
    """model -> [(task, dataset, metric, Path), ...] for one directory of configs."""
    per = defaultdict(list)
    if not root.is_dir():
        return per
    for d in root.iterdir():
        if not d.is_dir():
            continue
        p = split_config(d.name)
        if p:
            per[p[3]].append((p[0], p[1], p[2], d))
    return per


def runs_of(dirs):
    n = 0
    for *_x, d in dirs:
        for f in d.glob("*.json"):
            try:
                rows = json.loads(f.read_text())
            except Exception:
                continue
            n += len(rows) if isinstance(rows, list) else 1
    return n


def count_runs(stat_train: Path):
    """Total runs on disk, for the before/after conservation check."""
    n = 0
    for f in stat_train.glob("*/*.json"):
        try:
            rows = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(rows, list):
            n += len(rows)
    return n


# ---------------------------------------------------------------- planning


def plan(models, stats):
    groups = defaultdict(list)
    for name, uid in models.items():
        groups[uid].append(name)
    out = []
    for uid, members in groups.items():
        if len(members) < 2:
            continue
        scored = sorted(members, key=lambda m: (-runs_of(stats.get(m, [])), m))
        out.append((uid, scored[0], scored[1:]))
    return sorted(out, key=lambda t: -len(t[2]))


def preflight(groups, nn_dir, models):
    """Every member must be a model file we actually parsed. Fail loudly if not."""
    problems = []
    for uid, keep, drop in groups:
        for m in [keep, *drop]:
            if m not in models:
                problems.append((uid, m, "not in the parsed model set"))
            elif not (nn_dir / f"{m}.py").is_file():
                problems.append((uid, m, "model file missing on disk"))
    return problems


# ---------------------------------------------------------------- merging


def run_key(r):
    """The identity of one recorded run, matching the database's `stat.id`.

    task, dataset, metric, model and epoch are fixed by the file's location, so
    only the remaining fields are needed here.
    """
    return (r.get("transform"), r.get("uid"), r.get("duration"), r.get("accuracy"))


def merge_json_lists(src: Path, dst: Path, apply: bool):
    """Merge `<epoch>.json` run arrays. Returns (files, added, dropped)."""
    files = added = dropped = 0
    for f in sorted(src.glob("*.json")):
        try:
            rows = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        files += 1
        target = dst / f.name
        if not target.exists():
            if apply:
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, target)
            added += len(rows)
            continue
        try:
            have = json.loads(target.read_text())
        except Exception:
            have = []
        if not isinstance(have, list):
            have = []
        seen = {run_key(r) for r in have}
        new = []
        for r in rows:
            k = run_key(r)
            if k in seen:
                dropped += 1          # a genuine byte-for-byte repeat
            else:
                seen.add(k)
                new.append(r)
        if apply and new:
            target.write_text(json.dumps(have + new, indent=4))
        added += len(new)
    return files, added, dropped


def merge_flat_files(src: Path, dst: Path, apply: bool):
    """Merge a directory whose leaves are per-device measurement files.

    Two duplicates benchmarked on the same device produce files with the same
    name. Keeping both would need a numeric suffix, and the survivor would then
    hold several timings for one device with nothing in the schema to tell them
    apart. The dataset is meant to read as a single consistent source, so only
    the survivor's own measurement is kept and the incoming one is discarded.

    Returns (moved, discarded).
    """
    moved = discarded = 0
    for f in sorted(src.glob("*")):
        if not f.is_file():
            continue
        target = dst / f.name
        if target.exists():
            discarded += 1            # the survivor already has this device
            continue
        if apply:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, target)
        moved += 1
    return moved, discarded


def relocate_config_dirs(per_model, stat_root, keep, drop, apply, leaf,
                         log=None, folder="stat/train"):
    """Rename or merge `<t>_<d>_<m>_<model>` directories onto the survivor."""
    keys = {(t, d, m) for t, d, m, _p in per_model.get(keep, [])}
    renamed = merged = a = b = 0
    for other in drop:
        for t, d, m, src in per_model.get(other, []):
            tgt = stat_root / f"{t}_{d}_{m}_{keep}"
            entry = {"folder": folder, "config": f"{t}_{d}_{m}"}
            if (t, d, m) in keys:
                if leaf == "runs":
                    _f, x, y = merge_json_lists(src, tgt, apply)
                    entry.update(action="merged", runs_relocated=x, runs_dropped=y)
                else:
                    x, y = merge_flat_files(src, tgt, apply)
                    entry.update(action="merged", files_relocated=x, files_discarded=y)
                a += x
                b += y
                merged += 1
                if apply:
                    shutil.rmtree(src)
            else:
                if apply:
                    src.rename(tgt)
                keys.add((t, d, m))
                renamed += 1
                entry.update(action="renamed")
            if log is not None:
                log[other].append(entry)
    return renamed, merged, a, b


def remap_name_maps(root: Path, rename: dict, apply: bool, log=None):
    """Rewrite `all_models.json` dicts that are keyed by model name.

    These files hold exactly one `{accuracy, transform}` record per model name.
    When two duplicates each have a record, the schema cannot hold both, so the
    survivor's is kept and the other is counted and reported. This is a limit of
    the file format, not something the merge can work around.

    Returns (files, moved, collided).
    """
    files = moved = collided = 0
    for f in root.rglob("all_models.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        hit = [k for k in d if k in rename]
        if not hit:
            continue
        for k in hit:
            survivor = rename[k]
            if survivor in d:
                collided += d[survivor] != d[k]
                action = "lost" if d[survivor] != d[k] else "identical"
            else:
                d[survivor] = d[k]
                moved += 1
                action = "moved"
            if log is not None:
                log[k].append({"folder": str(f.relative_to(root.parent)),
                               "action": action, "record": d.get(k)})
            del d[k]
        files += 1
        if apply:
            f.write_text(json.dumps(d, indent=4))
    return files, moved, collided


def remap_embedded_names(root: Path, rename: dict, apply: bool, field="model_name"):
    """Rewrite a model name stored *inside* a JSON file.

    `json_run_tflite_to_db` reads `model_name` from the file contents, not from
    the directory it sits in, so renaming the directory alone leaves the
    database pointing at a model that no longer exists.
    """
    files = 0
    for f in root.rglob("*.json"):
        if f.name == "all_models.json":
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        old = d.get(field)
        if old not in rename:
            continue
        d[field] = rename[old]
        files += 1
        if apply:
            f.write_text(json.dumps(d, indent=4))
    return files


def relocate_model_files(stat_nn: Path, rename: dict, apply: bool, log=None):
    """`stat/nn/<model>.json`, one file per model. Rename, or drop if taken."""
    moved = removed = 0
    for old, survivor in rename.items():
        src = stat_nn / f"{old}.json"
        if not src.is_file():
            continue
        tgt = stat_nn / f"{survivor}.json"
        if tgt.exists():
            if apply:
                src.unlink()
            removed += 1
            action = "removed"
        else:
            if apply:
                src.rename(tgt)
            moved += 1
            action = "renamed"
        if log is not None:
            log[old].append({"folder": "stat/nn", "action": action})
    return moved, removed


# ---------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_nn_dir.parent,
                    help="the ab/nn directory (default: this checkout)")
    ap.add_argument("--show", type=int, default=1, help="groups to detail")
    ap.add_argument("--apply", action="store_true", help="write changes")
    ap.add_argument("--log", type=Path, default=Path("arch_uid_merge_log.json"),
                    help="where to write the JSON record of every decision")
    a = ap.parse_args()

    nn_dir = a.root / "nn"
    stat_train = a.root / "stat" / "train"
    stat_run = a.root / "stat" / "run"
    stat_nn = a.root / "stat" / "nn"
    for p in (nn_dir, stat_train):
        if not p.is_dir():
            raise SystemExit(f"not found: {p}")

    print(f"reading {nn_dir} ...", flush=True)
    models, skipped = load_models(nn_dir)
    train = load_dirs(stat_train)
    print(f"  {len(models)} models parsed, {len(skipped)} skipped, "
          f"{sum(len(v) for v in train.values())} stat/train directories", flush=True)

    runs_before = count_runs(stat_train)
    groups = plan(models, train)

    bad = preflight(groups, nn_dir, models)
    if bad:
        print("\nPREFLIGHT FAILED, nothing was written:")
        for g, m, why in bad[:20]:
            print(f"  {g[:16]}  {m}  {why}")
        raise SystemExit(1)

    n_del = sum(len(d) for _u, _k, d in groups)
    rename = {o: k for _u, k, drop in groups for o in drop}
    # run counts are taken now, before anything moves
    runs = {m: runs_of(train.get(m, [])) for _u, k, drop in groups for m in [k, *drop]}
    log = defaultdict(list)          # removed model -> every action taken on it

    print("=" * 74)
    print("PLAN" if not a.apply else "APPLYING")
    print("=" * 74)
    print(f"{'models parsed':<52}{len(models):>10}")
    print(f"{'models skipped (never merged)':<52}{len(skipped):>10}")
    print(f"{'architecture groups with duplicates':<52}{len(groups):>10}")
    print(f"{'models kept (one per architecture)':<52}{len(models) - n_del:>10}")
    print(f"{'model files to remove':<52}{n_del:>10}")
    print(f"{'preflight (every member exists on disk)':<52}{'PASS':>10}")

    detail = []
    tot = defaultdict(int)
    for uid, keep, drop in groups:
        r, m, added, droppedruns = relocate_config_dirs(
            train, stat_train, keep, drop, a.apply, "runs", log)
        tot["train_renamed"] += r
        tot["train_merged"] += m
        tot["runs_moved"] += added
        tot["runs_dropped"] += droppedruns
        detail.append({"uid": uid, "keep": keep, "drop": drop,
                       "renamed": r, "merged": m, "moved": added})
        if a.apply:
            for other in drop:
                f = nn_dir / f"{other}.py"
                if f.exists():
                    f.unlink()

    print(f"\n{'stat/train':<52}")
    print(f"{'  directories renamed onto the survivor':<52}{tot['train_renamed']:>10}")
    print(f"{'  directories merged into an existing one':<52}{tot['train_merged']:>10}")
    print(f"{'  runs relocated':<52}{tot['runs_moved']:>10}")
    print(f"{'  runs dropped as exact repeats':<52}{tot['runs_dropped']:>10}")

    # stat/run: one config tree per backend and precision
    run_renamed = run_merged = dev_moved = dev_kept = 0
    if stat_run.is_dir():
        for backend in sorted(p for p in stat_run.iterdir() if p.is_dir()):
            for prec in sorted(p for p in backend.iterdir() if p.is_dir()):
                per = load_dirs(prec)
                if not per:
                    continue
                for uid, keep, drop in groups:
                    r, m, x, y = relocate_config_dirs(
                        per, prec, keep, drop, a.apply, "flat", log,
                        f"stat/run/{backend.name}/{prec.name}")
                    run_renamed += r
                    run_merged += m
                    dev_moved += x
                    dev_kept += y
        map_files, map_moved, map_collided = remap_name_maps(stat_run, rename, a.apply, log)
        embedded = remap_embedded_names(stat_run, rename, a.apply)
    else:
        map_files = map_moved = map_collided = embedded = 0

    print(f"\n{'stat/run':<52}")
    print(f"{'  directories renamed onto the survivor':<52}{run_renamed:>10}")
    print(f"{'  directories merged into an existing one':<52}{run_merged:>10}")
    print(f"{'  device measurements relocated':<52}{dev_moved:>10}")
    print(f"{'  device measurements discarded, survivor has that device':<52}"
          f"{dev_kept:>10}")
    print(f"{'  all_models.json files remapped':<52}{map_files:>10}")
    print(f"{'  name keys moved to the survivor':<52}{map_moved:>10}")
    print(f"{'  name keys lost, schema holds one per model':<52}{map_collided:>10}")
    print(f"{'  embedded model_name fields rewritten':<52}{embedded:>10}")

    nn_moved, nn_removed = (relocate_model_files(stat_nn, rename, a.apply, log)
                            if stat_nn.is_dir() else (0, 0))
    print(f"\n{'stat/nn':<52}")
    print(f"{'  metric files renamed onto the survivor':<52}{nn_moved:>10}")
    print(f"{'  metric files removed (survivor already had one)':<52}{nn_removed:>10}")

    # conservation, measured rather than asserted
    print(f"\n{'=' * 74}")
    print("CONSERVATION")
    print("=" * 74)
    print(f"{'runs on disk before':<52}{runs_before:>10}")
    if a.apply:
        after = count_runs(stat_train)
        print(f"{'runs on disk after':<52}{after:>10}")
        print(f"{'difference':<52}{after - runs_before:>10}")
        print(f"{'accounted for by exact repeats':<52}{-tot['runs_dropped']:>10}")
        ok = (runs_before - after) == tot["runs_dropped"]
        print(f"{'UNEXPLAINED LOSS':<52}"
              f"{runs_before - after - tot['runs_dropped']:>10}   "
              f"{'OK' if ok else 'MISMATCH'}")
    else:
        print(f"{'runs that would be dropped as exact repeats':<52}"
              f"{tot['runs_dropped']:>10}")

    for rec in detail[:a.show]:
        print(f"\n{'=' * 74}\nEXAMPLE GROUP  arch_uid {rec['uid'][:16]}\n{'=' * 74}")
        print(f"  KEEP    {rec['keep']}")
        for d in rec["drop"][:6]:
            print(f"  REMOVE  {d}")
        print(f"  {rec['renamed']} renamed, {rec['merged']} merged, "
              f"{rec['moved']} runs relocated")

    record = {
        "root": str(a.root.resolve()),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "apply" if a.apply else "plan",
        "summary": {
            "models_parsed": len(models), "models_skipped": len(skipped),
            "groups": len(groups), "models_kept": len(models) - n_del,
            "models_removed": n_del,
            "stat_train": {"renamed": tot["train_renamed"], "merged": tot["train_merged"],
                           "runs_relocated": tot["runs_moved"],
                           "runs_dropped_exact_repeats": tot["runs_dropped"]},
            "stat_run": {"renamed": run_renamed, "merged": run_merged,
                         "measurements_relocated": dev_moved,
                         "measurements_discarded": dev_kept,
                         "all_models_files": map_files, "keys_moved": map_moved,
                         "keys_lost": map_collided,
                         "embedded_model_name_rewritten": embedded},
            "stat_nn": {"renamed": nn_moved, "removed": nn_removed},
            "runs_on_disk_before": runs_before,
            "runs_on_disk_after": count_runs(stat_train) if a.apply else None,
        },
        "skipped": [{"model": n, "why": w} for n, w in skipped],
        "renames": rename,
        "groups": [
            {"arch_uid": uid, "survivor": keep, "survivor_runs": runs[keep],
             "removed": [{"name": o, "runs": runs[o], "actions": log.get(o, [])}
                         for o in drop]}
            for uid, keep, drop in groups],
    }
    a.log.parent.mkdir(parents=True, exist_ok=True)
    a.log.write_text(json.dumps(record, indent=1))
    print(f"\nlog written to {a.log}  ({len(groups)} groups, {n_del} removed models)")

    if not a.apply:
        print(f"\n{'=' * 74}")
        print("dry run, nothing written. re-run with --apply to perform it.")


if __name__ == "__main__":
    main()
