"""Attach BLIP-2 run provenance to parameters saved by the existing trainer."""

import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

from .contract import read_manifest, resolve_cache_dir, sha256_file


def record_training_provenance(model, prm, decoder):
    root = resolve_cache_dir(model.prm.get("cache_dir"))
    manifest = read_manifest(root)
    repo = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    sources = list((repo / "ab/nn/captioning/blip2").glob("*.py"))
    sources += [repo / "ab/nn" / name for name in (
        "nn/Blip2Fast.py", "nn/Blip2FastOpt.py", "nn/Blip2Cached.py",
        "metric/caption_text.py", "metric/bleu.py", "metric/meteor.py", "metric/cider.py",
        "tools/build_blip2_cached.py", "tools/prepare_blip2_gpt2_runtime.py", "util/Train.py",
    )]
    limits = {}
    for split, record in manifest.get("splits", {}).items():
        available = sum(int(shard["samples"]) for shard in record.get("shards", []))
        limit = int(os.environ.get(f"BLIP2_{split.upper()}_LIMIT", "0"))
        limits[split] = {"cached_samples": available, "effective_samples": min(limit, available) if limit else available}
    generation = decoder.generation_config.to_dict()
    tokenizer = getattr(model, "opt_tokenizer", None) or model.gpt2_tokenizer
    generation.update(max_new_tokens=model.max_new_tokens, num_beams=model.num_beams,
                      eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    record = {
        "code_commit": commit,
        "source_sha256": {str(path.relative_to(repo)): sha256_file(path) for path in sorted(sources)},
        "cache_manifest_sha256": sha256_file(root / "manifest.json"),
        "source_checkpoint": {"model_id": manifest["model_id"], "revision": manifest["model_revision"],
                              "runtime": manifest.get("runtime"), "gpt2_runtime": manifest.get("gpt2_runtime")},
        "dataset": limits,
        "seed": int(model.prm.get("seed", 42)),
        "prompt": model.prompt,
        "max_text_length": model.max_text_length,
        "generation": generation,
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "transformers", "nltk", "numpy", "pillow")},
        "metrics": {"bleu": "NLTK sentence BLEU-4, smoothing method1, mean",
                    "meteor": "NLTK METEOR with WordNet; empty hypotheses score zero",
                    "cider": "internal CIDEr approximation divided by 3 and capped at 1"},
    }
    projection = getattr(model, "projection_path", None)
    if projection is not None:
        record["initial_projection_sha256"] = sha256_file(projection)
    # Train.py already saves this dict; no core trainer or cache writes needed.
    prm["captioning_provenance"] = record
    model.prm["captioning_provenance"] = record
