"""Top up existing stat files with nn_depth, without recomputing anything else.

`log_nn_stat --rewrite` recomputes every metric and, if a model now fails to
instantiate (missing dataset, OOM, API drift), replaces its stat file with an
error record -- so a full rewrite can lose statistics that are currently fine.

This script is non-destructive and resumable: it reads each existing
ab/nn/stat/nn/*.json, computes only nn_depth, and merges it back. Files that
already carry nn_depth are skipped, as are error records.

When nn_depth cannot be measured the field is left out of the file entirely
rather than stored as a sentinel; the model keeps all its other statistics and
is listed at the end. A stale nn_depth of -1 written by an earlier version is
treated as absent and removed if it still cannot be measured.

    python -m util.py.add_nn_depth                  # all models
    python -m util.py.add_nn_depth --nn ResNet      # one model
    python -m util.py.add_nn_depth --limit 50       # smoke test
    python -m util.py.add_nn_depth --redo           # recompute nn_depth too
"""
import argparse
import json

import torch

from ab.nn.api import data
from ab.nn.util.Const import stat_nn_dir
from ab.nn.util.Loader import load_dataset
from ab.nn.util.NNAnalysis import get_nn_depth
from ab.nn.util.Util import get_in_shape, torch_device, first_tensor


def main():
    ap = argparse.ArgumentParser(description="Add nn_depth to existing LEMUR stat files.")
    ap.add_argument('--nn', type=str, default=None, help="Filter by neural network name")
    ap.add_argument('--limit', type=int, default=None, help="Limit the number of models to process")
    ap.add_argument('--redo', action='store_true',
                    help="Recompute nn_depth even where it is already present")
    args = ap.parse_args()

    df = data(nn=args.nn, max_rows=args.limit).drop_duplicates(subset=["nn"], keep="first")
    done = skipped = failed = no_stat = unmeasured = 0
    failures, unmeasurable = [], []

    for i, (_, row) in enumerate(df.iterrows(), 1):
        nn = row["nn"]
        f_nm = stat_nn_dir / f"{nn}.json"

        if not f_nm.exists():
            no_stat += 1
            continue
        with f_nm.open(encoding="utf-8") as f:
            stats = json.load(f)
        if "error" in stats:
            skipped += 1
            continue
        # -1 was the old sentinel; treat it as absent so it gets retried
        was_stale = stats.get("nn_depth") == -1
        if was_stale:
            stats.pop("nn_depth")
        if "nn_depth" in stats and not args.redo:
            skipped += 1
            continue

        try:
            prm = row["prm"]
            if isinstance(prm, str):
                prm = json.loads(prm.replace("'", '"'))

            local_scope = {"torch": torch, "nn": torch.nn}
            exec(row["nn_code"], local_scope, local_scope)

            out_shape, _, train_set, _ = load_dataset(row["task"], row["dataset"], prm["transform"])
            input_tensor = first_tensor(train_set)
            in_shape = get_in_shape(train_set)

            model = local_scope["Net"](in_shape, out_shape, prm, torch_device()).to(torch_device())
            depth = get_nn_depth(model, input_tensor)
        except Exception as e:
            failed += 1
            failures.append((nn, repr(e)[:120]))
            print(f"{i}. {nn}: could not instantiate, statistics left unchanged")
            continue

        if depth is None:
            # not measurable: leave the field out instead of storing a sentinel
            unmeasured += 1
            unmeasurable.append(nn)
            if was_stale:
                with f_nm.open("w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=4, ensure_ascii=False)
                print(f"{i}. {nn}: not measurable, stale nn_depth removed")
            else:
                print(f"{i}. {nn}: not measurable, nn_depth left out")
            continue

        # rebuild the dict so nn_depth sits next to max_depth rather than at the end
        new = {}
        for k, v in stats.items():
            new[k] = v
            if k == "max_depth":
                new["nn_depth"] = depth
        if "nn_depth" not in new:
            new["nn_depth"] = depth

        with f_nm.open("w", encoding="utf-8") as f:
            json.dump(new, f, indent=4, ensure_ascii=False)
        done += 1
        print(f"{i}. {nn}: nn_depth = {depth}")

    print(f"\nupdated {done}, skipped {skipped}, not measurable {unmeasured}, "
          f"no stat file {no_stat}, could not instantiate {failed}")
    if unmeasurable:
        print(f"nn_depth omitted for {len(unmeasurable)} models "
              f"(all other statistics intact): {', '.join(unmeasurable[:10])}"
              f"{' ...' if len(unmeasurable) > 10 else ''}")
    if failures:
        print("could not instantiate (statistics preserved):")
        for nn, err in failures[:20]:
            print(f"  {nn}: {err}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
