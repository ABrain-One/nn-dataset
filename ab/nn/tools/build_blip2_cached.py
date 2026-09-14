"""Resumable builder for the clean BLIP-2 Q-Former feature cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ab.nn.captioning.blip2.environment import validate_environment

validate_environment()

from transformers import AutoProcessor, Blip2ForConditionalGeneration

from ab.nn.captioning.blip2.contract import (
    CACHE_VERSION, FEATURE_DTYPE, FEATURE_SHAPE, MANIFEST_NAME, MODEL_ID, MODEL_REVISION,
    OPT_DIR_NAME, OPT_TOKENIZER_DIR_NAME, PROJECTION_NAME, RUNTIME_DIR_NAME, atomic_json,
    resolve_cache_dir, sha256_file,
)


class CocoImages(Dataset):
    def __init__(self, root: Path, split: str):
        self.image_dir = root / f"{split}2017"
        annotation = root / "annotations" / f"captions_{split}2017.json"
        if not annotation.is_file() or not self.image_dir.is_dir():
            raise FileNotFoundError(f"COCO {split} files are incomplete under {root}")
        self.coco = COCO(str(annotation))
        self.ids = sorted(self.coco.imgs)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        info = self.coco.loadImgs(image_id)[0]
        path = self.image_dir / info["file_name"]
        with Image.open(path) as image:
            rgb = image.convert("RGB").copy()
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        captions = [item["caption"] for item in annotations if item.get("caption", "").strip()]
        if not captions:
            raise RuntimeError(f"COCO image {image_id} has no captions")
        return rgb, captions, image_id


def _collate(batch):
    return [item[0] for item in batch], [item[1] for item in batch], [item[2] for item in batch]


def _manifest(cache_dir):
    path = cache_dir / MANIFEST_NAME
    if path.is_file():
        import json
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "cache_version": CACHE_VERSION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "feature_shape": list(FEATURE_SHAPE),
            "feature_dtype": FEATURE_DTYPE,
        }
        mismatches = [
            name for name, expected_value in expected.items()
            if value.get(name) != expected_value
        ]
        if mismatches:
            raise RuntimeError(
                "Existing cache has an incompatible contract: "
                + ", ".join(mismatches)
            )
        return value
    return {
        "cache_version": CACHE_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "feature_shape": list(FEATURE_SHAPE),
        "feature_dtype": FEATURE_DTYPE,
        "projection": {},
        "splits": {},
    }


def _export_runtime(cache_dir, model, processor, manifest):
    runtime_dir = cache_dir / RUNTIME_DIR_NAME
    opt_dir = runtime_dir / OPT_DIR_NAME
    opt_tokenizer_dir = runtime_dir / OPT_TOKENIZER_DIR_NAME
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if not (opt_dir / "config.json").is_file():
        model.language_model.save_pretrained(opt_dir, safe_serialization=True)
    if not (opt_tokenizer_dir / "tokenizer_config.json").is_file():
        processor.tokenizer.save_pretrained(opt_tokenizer_dir)

    records = []
    for path in sorted(runtime_dir.rglob("*")):
        if path.is_file():
            records.append({
                "path": str(path.relative_to(runtime_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    manifest["runtime"] = {"complete": True, "files": records}
    atomic_json(cache_dir / MANIFEST_NAME, manifest)


def build(
    root: Path,
    cache_dir: Path,
    split: str,
    batch_size: int,
    shard_size: int,
    allow_cpu: bool = False,
    limit: int | None = None,
):
    if batch_size < 1 or shard_size < 1:
        raise ValueError("batch-size and shard-size must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(cache_dir)
    existing = manifest.get("splits", {}).get(split, {})
    completed = sum(int(item["samples"]) for item in existing.get("shards", []))
    dataset = CocoImages(root, split)
    target_size = len(dataset) if limit is None else min(len(dataset), limit)
    if completed > target_size:
        raise RuntimeError(
            f"Cache already contains {completed} {split} samples, which exceeds "
            f"the requested target of {target_size}. Use another --cache-dir."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError(
            "Cache extraction refused to load full BLIP-2 on CPU because it can "
            "exhaust system RAM. Use CUDA or pass --allow-cpu explicitly."
        )
    if device.type == "cuda":
        total_gib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if total_gib < 12:
            raise RuntimeError(
                f"Cache extraction requires at least 12 GiB VRAM; detected {total_gib:.1f} GiB. "
                "Build the cache on a larger GPU and copy the validated cache."
            )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()

    _export_runtime(cache_dir, model, processor, manifest)

    projection_path = cache_dir / PROJECTION_NAME
    if not projection_path.is_file():
        temporary = projection_path.with_suffix(f".tmp.{os.getpid()}")
        torch.save({key: value.detach().cpu() for key, value in model.language_projection.state_dict().items()}, temporary)
        os.replace(temporary, projection_path)
    manifest["projection"] = {"filename": PROJECTION_NAME, "sha256": sha256_file(projection_path)}

    subset = torch.utils.data.Subset(dataset, range(completed, target_size))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
    records = list(existing.get("shards", []))
    feature_buffer, caption_buffer, id_buffer = [], [], []
    shard_index = len(records)

    def flush():
        nonlocal feature_buffer, caption_buffer, id_buffer, shard_index
        if not feature_buffer:
            return
        combined = torch.cat(feature_buffer)
        features = combined[:shard_size]
        remainder = combined[shard_size:]
        count = len(features)
        captions, ids = caption_buffer[:count], id_buffer[:count]
        filename = f"{split}-{shard_index:05d}.pt"
        destination = cache_dir / filename
        temporary = destination.with_suffix(f".tmp.{os.getpid()}")
        torch.save({"features": features, "captions": captions, "image_ids": ids}, temporary)
        os.replace(temporary, destination)
        records.append({"filename": filename, "samples": count, "size_bytes": destination.stat().st_size, "sha256": sha256_file(destination)})
        manifest["splits"][split] = {"complete": False, "samples": sum(r["samples"] for r in records), "shards": records}
        atomic_json(cache_dir / MANIFEST_NAME, manifest)
        feature_buffer = [remainder] if len(remainder) else []
        caption_buffer, id_buffer = caption_buffer[count:], id_buffer[count:]
        shard_index += 1

    with torch.inference_mode():
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"Extracting BLIP-2 {split} features",
            unit="batch",
            dynamic_ncols=True,
        )
        for images, captions, image_ids in progress:
            inputs = processor(images=images, return_tensors="pt")
            pixels = inputs.pixel_values.to(device=device, dtype=dtype)
            vision = model.vision_model(pixel_values=pixels, return_dict=True).last_hidden_state
            vision_mask = torch.ones(vision.shape[:-1], dtype=torch.long, device=device)
            queries = model.query_tokens.expand(vision.shape[0], -1, -1)
            features = model.qformer(
                query_embeds=queries,
                encoder_hidden_states=vision,
                encoder_attention_mask=vision_mask,
                return_dict=True,
            ).last_hidden_state.cpu().half()
            feature_buffer.append(features)
            caption_buffer.extend(captions)
            id_buffer.extend(image_ids)
            while sum(len(value) for value in feature_buffer) >= shard_size:
                flush()
            progress.set_postfix(
                cached=sum(int(record["samples"]) for record in records),
                shards=len(records),
                refresh=False,
            )
        flush()
    manifest["splits"][split] = {
        "complete": True,
        "samples": target_size,
        "source_samples": len(dataset),
        "limited": target_size != len(dataset),
        "shards": records,
    }
    atomic_json(cache_dir / MANIFEST_NAME, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Build only the first N samples (use a separate cache directory).",
    )
    args = parser.parse_args()
    build(
        args.coco_root.expanduser().resolve(),
        resolve_cache_dir(args.cache_dir),
        args.split,
        args.batch_size,
        args.shard_size,
        args.allow_cpu,
        args.limit,
    )


if __name__ == "__main__":
    main()
