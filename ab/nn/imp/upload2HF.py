#!/usr/bin/env python3
"""
Upload converted TFLite models and accuracy stats to HuggingFace (optional).

Quantization modes (positional argument):
  all      - FP32, FP16, mixed, static INT8, then dynamic INT8 (default)
  fp32     - FP32 only
  fp16     - post-training FP16 only
  mixed    - post-training mixed precision (calibrated INT8 ops + float fallback)
  static   - static INT8 only
  dynamic  - dynamic INT8 only

Examples:
  python ab/nn/imp/upload2HF.py
  python ab/nn/imp/upload2HF.py mixed
  python ab/nn/imp/upload2HF.py fp16
  python ab/nn/imp/upload2HF.py dynamic
  python ab/nn/imp/upload2HF.py all --push-hf --hf-token hf_xxx
  # ImageNet-100 (same pipeline; separate HF prefix + local stats):
  python ab/nn/imp/upload2HF.py fp32 --dataset imagenet100 --limit-models 1
  python ab/nn/imp/upload2HF.py all --dataset imagenet100 --push-hf
"""
import sys
import os
import argparse
import json
import re
import subprocess
import importlib.util
import shutil
import time
import gc
from pathlib import Path
from typing import Any, Dict

# --- CONFIGURATION ---
TARGET_REPO = "NN-Dataset/tflite"
DEFAULT_HF_TOKEN = ""  # Set via --hf-token or HF_TOKEN env var or paste here

# Dataset-specific defaults (CIFAR path preserved; ImageNet-100 added).
DATASET_CONFIGS = {
    "cifar-10": {
        "source_repo": "NN-Dataset/checkpoints-epoch-50",
        "hf_prefix": "img-classification_cifar-10_acc",
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "default_h": 32,
        "eval_name": "CIFAR-10",
        "stats_suffix": "",
    },
    "imagenet100": {
        # Matches the active training upload repo.
        "source_repo": "NN-Dataset/checkpoints-imagenet100-epoch-50",
        "hf_prefix": "img-classification_imagenet100_acc",
        "num_classes": 100,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "default_h": 160,
        "eval_name": "ImageNet-100",
        "stats_suffix": "_imagenet100",
    },
}
# ---------------------

# --- 1. SETUP PATHS ---
script_path = Path(__file__).resolve()
dataset_root = script_path.parents[3]
CONVERT_WORKER = script_path.parent / "tflite_convert_worker.py"

if str(dataset_root) not in sys.path:
    sys.path.insert(0, str(dataset_root))

# --- WORK DIRS ---
work_dir = dataset_root / "_work"
out_dir = work_dir / "stats"       
data_root = work_dir / "data"      
temp_dl_dir = work_dir / "temp"    

HISTORY_FILE = out_dir / "upload_history.json"
SKIPPED_FILE = out_dir / "skipped_models.json"
FAILED_FILE = out_dir / "upload_failed.json"

HISTORY_FILES_BASE = {
    "all": "upload_history.json",
    "fp32": "upload_history_fp32.json",
    "fp16": "upload_history_fp16.json",
    "mixed": "upload_history_mixed.json",
    "static": "upload_history_static.json",
    "dynamic": "upload_history_dynamic.json",
}

for p in [out_dir, data_root, temp_dl_dir]:
    p.mkdir(parents=True, exist_ok=True)

# --- 2. IMPORTS ---
import torch
import torchvision
import torchvision.transforms as T
from huggingface_hub import hf_hub_download, list_repo_files, upload_file, create_repo

# ------------------------
# SMART UPLOAD FUNCTION
# ------------------------
def upload_with_retry(file_path, repo_path, repo_id):
    while True:
        try:
            upload_file(path_or_fileobj=str(file_path), path_in_repo=repo_path, repo_id=repo_id)
            return True 
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "too many requests" in error_msg:
                print(f"\n[WARN] 🛑 Rate Limit Hit! Sleeping for 65 minutes...")
                time.sleep(65 * 60) 
                continue 
            else:
                raise e

# ------------------------
# LOGGING HELPERS
# ------------------------
def load_json_safe(path: Path):
    if not path.exists() or path.stat().st_size == 0: return {}
    try:
        with open(path, "r") as f: return json.load(f)
    except: return {}

def mark_as_done(name: str, history_file: Path = HISTORY_FILE):
    current_list = []
    if history_file.exists() and history_file.stat().st_size > 0:
        try:
            with open(history_file, "r") as f: current_list = json.load(f)
        except: current_list = []
    
    if name not in current_list:
        current_list.append(name)
        with open(history_file, "w") as f:
            json.dump(current_list, f, indent=2)

def log_skip(name: str, reason: str):
    current_skips = load_json_safe(SKIPPED_FILE)
    if not any(d.get("model") == name for d in current_skips):
        current_skips.append({"model": name, "reason": reason})
        with open(SKIPPED_FILE, "w") as f:
            json.dump(current_skips, f, indent=2)
    print(f"   [SKIP] {reason}")

def get_resolution_from_transform_file(transform_name: str, transforms_dir: Path, default_h: int = 32) -> int:
    if not transform_name:
        print(f"   [DEBUG] No transform name. Defaulting to {default_h}.")
        return default_h
    for ext in [".py", ".json"]:
        f = transforms_dir / f"{transform_name}{ext}"
        if f.exists():
            print(f"   [DEBUG] Found transform file: {f.name}")
            try:
                content = f.read_text()
                match = re.search(r"(?:Resize|size|Crop).*?(\d+)", content, re.IGNORECASE)
                if match:
                    res = int(match.group(1))
                    print(f"   [DEBUG] Extracted resolution: {res}x{res}")
                    return res
            except Exception:
                pass
    print(f"   [DEBUG] Transform file '{transform_name}' NOT FOUND. Defaulting to {default_h}.")
    return default_h


def slug(s: str) -> str: return re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())[:200]

def log_fail(name: str, mode: str, reason: str):
    current = load_json_safe(FAILED_FILE)
    if not isinstance(current, list):
        current = []
    current.append({
        "model": name,
        "mode": mode,
        "reason": reason[:500],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(FAILED_FILE, "w") as f:
        json.dump(current, f, indent=2)

def eval_tflite_acc_subprocess(
    tflite_path: Path,
    data_root: Path,
    ds_cfg: Dict[str, Any],
    batch_size: int = 100,
) -> Dict[str, Any]:
    payload = {
        "tflite_path": str(tflite_path),
        "data_root": str(data_root),
        "batch_size": int(batch_size),
        "limit": 1000,
        "dataset": ds_cfg.get("_name", "cifar-10"),
        "mean": list(ds_cfg["mean"]),
        "std": list(ds_cfg["std"]),
        "default_h": int(ds_cfg["default_h"]),
        "imagenet_root": ds_cfg.get("imagenet_root", ""),
    }
    code = r"""
import json, sys, os, numpy as np, tensorflow as tf, torchvision as tv, torch
from pathlib import Path
p = json.loads(sys.stdin.read())
try:
    interp = tf.lite.Interpreter(
        model_path=p["tflite_path"],
        num_threads=1,
        experimental_disable_delegate_clustering=True,
        experimental_default_delegate_latest_features=False,
    )
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]; out_det = interp.get_output_details()[0]
    in_idx, out_idx, in_dtype = in_det["index"], out_det["index"], in_det["dtype"]
    in_shape = in_det["shape"]
    spatial = [d for d in in_shape if d > 3]
    default_h = int(p.get("default_h", 32))
    h, w = (spatial[-2], spatial[-1]) if len(spatial) >= 2 else (default_h, default_h)
    mean = tuple(p.get("mean", (0.4914, 0.4822, 0.4465)))
    std = tuple(p.get("std", (0.2023, 0.1994, 0.2010)))
    tfm = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize((h, w), antialias=True),
        tv.transforms.Normalize(mean, std),
    ])
    dataset_name = p.get("dataset", "cifar-10")
    if dataset_name == "imagenet100":
        root = Path(p.get("imagenet_root") or "")
        val_dirs = sorted(root.glob("val.X*")) if root.exists() else []
        if not val_dirs:
            raise FileNotFoundError(f"ImageNet-100 val.X* not found under {root}")
        test = tv.datasets.ImageFolder(root=str(val_dirs[0]), transform=tfm)
    else:
        test = tv.datasets.CIFAR10(root=p["data_root"], train=False, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(test, batch_size=p["batch_size"], shuffle=False)
    is_nhwc = (in_shape[-1] == 3)
    correct, total = 0, 0
    for i_batch, (x, y) in enumerate(loader):
        if total >= p["limit"]: break
        x_np = x.numpy().astype(np.float32)
        if is_nhwc: x_np = np.transpose(x_np, (0, 2, 3, 1))
        if in_dtype == np.int8:
            s, zp = in_det.get("quantization", (1.0, 0))
            q = np.round(x_np / s + zp).clip(-128, 127).astype(np.int8)
        elif in_dtype == np.uint8:
            s, zp = in_det.get("quantization", (1.0, 0))
            q = np.round(x_np / s + zp).clip(0, 255).astype(np.uint8)
        else:
            q = x_np.astype(in_dtype)
        for i in range(len(q)):
            interp.set_tensor(in_idx, q[i:i+1]); interp.invoke()
            if np.argmax(interp.get_tensor(out_idx)) == y[i].item(): correct += 1
            total += 1
    print(json.dumps({
        "ok": True,
        "acc": float(correct) / total if total else 0.0,
        "in_dtype": str(in_dtype),
        "out_dtype": str(out_det["dtype"]),
    }))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))
"""
    eval_env = os.environ.copy()
    eval_env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=eval_env, start_new_session=True,
    )
    out, err = proc.communicate(input=json.dumps(payload))
    if proc.returncode not in (0, None):
        msg = f"eval subprocess aborted (exit {proc.returncode})"
        if err and err.strip():
            msg = f"{msg}: {err.strip()[-300:]}"
        return {"ok": False, "error": msg}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        tail = (out or err or "").strip()[-300:]
        return {"ok": False, "error": f"eval subprocess bad output: {tail}"}

def rep_dataset(target_h: int, data_root: Path, ds_cfg: Dict[str, Any]):
    def rep():
        mean = ds_cfg["mean"]
        std = ds_cfg["std"]
        tfm = T.Compose([
            T.ToTensor(),
            T.Resize((target_h, target_h), antialias=True),
            T.Normalize(mean, std),
        ])
        if ds_cfg.get("_name") == "imagenet100":
            from torchvision.datasets import ImageFolder
            root = Path(ds_cfg["imagenet_root"])
            train_dir = root / "train"
            if not train_dir.exists():
                # fall back to first train.X* class tree if merged train/ missing
                train_parts = sorted(root.glob("train.X*"))
                if not train_parts:
                    raise FileNotFoundError(f"ImageNet-100 train dir not found under {root}")
                train_dir = train_parts[0]
            d = ImageFolder(root=str(train_dir), transform=tfm)
        else:
            d = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=tfm)
        n = min(50, len(d))
        for j in range(n):
            yield [d[j][0].unsqueeze(0).numpy()]
    return rep

def file_size_kib(path):
    return Path(path).stat().st_size / 1024

def verify_tflite(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"TFLite export missing or empty: {path}")

def print_conversion_summary(label, out_path, elapsed_ms, acc, original_path=None, io_types=None, eval_name="CIFAR-10"):
    out_kib = file_size_kib(out_path)
    inp_dt = (io_types or {}).get("in_dtype", "unknown")
    out_dt = (io_types or {}).get("out_dtype", "unknown")
    if original_path and Path(original_path).exists():
        orig_kib = file_size_kib(original_path)
        ratio = out_kib / orig_kib if orig_kib else 0.0
        shrink = orig_kib / out_kib if out_kib else 0.0
        print(f"   Original model size:    {orig_kib:.2f} KiB")
        print(f"   Quantized model size:   {out_kib:.2f} KiB")
        print(f"   Quantization Ratio:     {ratio:.2f} ({shrink:.1f}x smaller)")
    else:
        print(f"   Model size:             {out_kib:.2f} KiB")
    print(f"   Input dtype:            {inp_dt}")
    print(f"   Output dtype:           {out_dt}")
    print(f"   {eval_name} accuracy:      {acc:.4f}")
    print(f"   Total time:             {elapsed_ms:.2f} ms")
    print(f"   -> Successfully converted {label} model.")

def run_tflite_convert_subprocess(job: Dict[str, Any], timeout_s: int = 900) -> None:
    """Fresh Python process for TFLite export (native crashes must not kill the batch)."""
    if not CONVERT_WORKER.exists():
        raise FileNotFoundError(f"Missing conversion worker: {CONVERT_WORKER}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.Popen(
        [sys.executable, str(CONVERT_WORKER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, start_new_session=True,
    )
    try:
        out, err = proc.communicate(input=json.dumps(job), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        raise RuntimeError(f"TFLite conversion timed out after {timeout_s}s")
    if proc.returncode not in (0, None):
        tail = (err or out or "").strip()[-400:]
        raise RuntimeError(f"TFLite conversion aborted (exit {proc.returncode}): {tail}")
    try:
        res = json.loads(out.strip().splitlines()[-1])
    except Exception:
        tail = (out or err or "").strip()[-400:]
        raise RuntimeError(f"TFLite conversion bad output: {tail}")
    if not res.get("ok"):
        raise RuntimeError(res.get("error", "unknown conversion error"))


def run_convert_and_eval(process_label, save_label, mode, out_path, data_root, convert_ctx, ds_cfg, original_path=None):
    print(f"   [PROCESS] {process_label}...")
    t0 = time.perf_counter()
    job = {
        "mode": mode,
        "out_path": str(out_path),
        "data_root": str(data_root),
        **convert_ctx,
    }
    run_tflite_convert_subprocess(job)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    verify_tflite(out_path)
    res = eval_tflite_acc_subprocess(out_path, data_root, ds_cfg)
    if not res.get("ok"):
        raise RuntimeError(f"Accuracy eval failed: {res.get('error', 'unknown error')}")
    acc = res.get("acc", 0.0)
    print_conversion_summary(
        save_label, out_path, elapsed_ms, acc, original_path,
        io_types={"in_dtype": res.get("in_dtype"), "out_dtype": res.get("out_dtype")},
        eval_name=ds_cfg.get("eval_name", "accuracy"),
    )
    return acc

def export_fp32_reference(convert_ctx: dict, ref_path: Path, data_root: Path) -> Path:
    print(f"   [BASELINE] FP32 reference not found, exporting for size comparison...")
    t0 = time.perf_counter()
    job = {
        "mode": "fp32",
        "out_path": str(ref_path),
        "data_root": str(data_root),
        **convert_ctx,
    }
    run_tflite_convert_subprocess(job)
    verify_tflite(ref_path)
    print(f"   Baseline size:          {file_size_kib(ref_path):.2f} KiB")
    print(f"   Baseline export time:   {(time.perf_counter() - t0) * 1000:.2f} ms")
    return ref_path

# ------------------------
# Main
# ------------------------
def main():
    arch_dir = dataset_root / "ab" / "nn" / "nn"
    transforms_dir = dataset_root / "ab" / "nn" / "transform"
    local_models_json = dataset_root / "all_models.json"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "quant_mode",
        nargs="?",
        default="all",
        choices=["all", "fp32", "fp16", "mixed", "static", "dynamic"],
        help="Quantization mode: all, fp32, fp16, mixed, static, or dynamic",
    )
    ap.add_argument(
        "--dataset",
        default="cifar-10",
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset / checkpoint source (cifar-10 or imagenet100)",
    )
    ap.add_argument("--source-repo", default=None, help="Override HF checkpoint repo")
    ap.add_argument("--target-repo", default=TARGET_REPO, help="HF TFLite destination repo")
    ap.add_argument("--push-hf", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if model is already in resume/history (needed to push after a local-only run)",
    )
    ap.add_argument("--hf-token", default=DEFAULT_HF_TOKEN)
    ap.add_argument("--format", choices=["tflite", "onnx"], default="tflite")
    ap.add_argument("--limit-models", type=int, default=0, help="Process only first N pending models (0=all)")
    ap.add_argument(
        "--model",
        default=None,
        help="Process only this model name (e.g. ResNet). Useful for smoke tests.",
    )
    args = ap.parse_args()
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    ds_cfg = dict(DATASET_CONFIGS[args.dataset])
    ds_cfg["_name"] = args.dataset
    source_repo = args.source_repo or ds_cfg["source_repo"]
    target_repo = args.target_repo
    suffix = ds_cfg["stats_suffix"]

    if args.dataset == "imagenet100":
        candidates = []
        try:
            from ab.nn.util.Const import data_dir
            candidates.append(data_dir / "imagenet100")
        except Exception:
            pass
        # Const often resolves to nn-dataset/; data may live at repo-parent/data/
        candidates.extend([
            dataset_root / "data" / "imagenet100",
            dataset_root.parent / "data" / "imagenet100",
        ])
        imagenet_root = next((p for p in candidates if p.exists()), candidates[0])
        ds_cfg["imagenet_root"] = str(imagenet_root)
        if not imagenet_root.exists():
            print(f"[WARN] ImageNet-100 data not found at {imagenet_root}")
            print("       Place val.X* / train (or train.X*) under that path before eval.")
        else:
            print(f"[CONFIG] imagenet_root={imagenet_root}")

    if args.format == "onnx":
        from onnx_pipeline import run_onnx_pipeline
        run_onnx_pipeline(args, dataset_root)
        return

    print(f"[CONFIG] dataset={args.dataset} | source={source_repo} | target={target_repo}")
    print(f"[CONFIG] hf_prefix={ds_cfg['hf_prefix']} | classes={ds_cfg['num_classes']} | default_h={ds_cfg['default_h']}")

    if args.push_hf:
        create_repo(target_repo, repo_type="model", exist_ok=True)

    try:
        downloaded_json = hf_hub_download(
            repo_id=source_repo, filename="all_models.json", local_dir=str(out_dir), force_download=True,
        )
        shutil.copy(downloaded_json, local_models_json)
    except Exception as e:
        print(f"[WARN] Could not refresh all_models.json from {source_repo}: {e}")

    with open(local_models_json) as f:
        model_db = json.load(f)

    json_fp32_path = out_dir / f"all_models_accuracy_fp32{suffix}.json"
    json_fp16_path = out_dir / f"all_models_accuracy_fp16{suffix}.json"
    json_mixed_path = out_dir / f"all_models_accuracy_mixed{suffix}.json"
    json_int8_path = out_dir / f"all_models_accuracy_int8{suffix}.json"
    json_dynamic_path = out_dir / f"all_models_accuracy_dynamic{suffix}.json"
    data_fp32 = load_json_safe(json_fp32_path)
    data_fp16 = load_json_safe(json_fp16_path)
    data_mixed = load_json_safe(json_mixed_path)
    data_int8 = load_json_safe(json_int8_path)
    data_dynamic = load_json_safe(json_dynamic_path)

    history_file = out_dir / HISTORY_FILES_BASE[args.quant_mode].replace(".json", f"{suffix}.json")
    hist_raw = load_json_safe(history_file)
    if isinstance(hist_raw, list):
        processed_history = set(hist_raw)
    elif isinstance(hist_raw, dict):
        processed_history = set(hist_raw.keys())
    else:
        processed_history = set()

    if args.quant_mode in ("all", "fp32"):
        processed_history.update(data_fp32.keys())
    elif args.quant_mode == "fp16":
        processed_history.update(data_fp16.keys())
    elif args.quant_mode == "mixed":
        processed_history.update(data_mixed.keys())
    elif args.quant_mode == "static":
        processed_history.update(data_int8.keys())
    elif args.quant_mode == "dynamic":
        processed_history.update(data_dynamic.keys())

    hf_files = list_repo_files(source_repo)
    py_files = sorted([p for p in arch_dir.rglob("*.py") if f"{p.stem}.pth" in hf_files])

    print(f"\n[{args.quant_mode.upper()} RUN] Models to process: {len(py_files)}")
    if args.model:
        want = slug(args.model)
        py_files = [p for p in py_files if slug(p.stem) == want]
        if not py_files:
            print(f"[ERROR] --model {args.model} not found among HF .pth + local .py")
            return
        print(f"[FILTER] --model {args.model} -> {len(py_files)} file(s)")
    if args.resume and not args.force:
        before = len(py_files)
        py_files = [p for p in py_files if slug(p.stem) not in processed_history]
        print(f"[RESUME] skip done={before - len(py_files)} | pending={len(py_files)}")
    elif args.force:
        print(f"[FORCE] reprocessing {len(py_files)} model(s) (resume ignored)")
    if args.limit_models and args.limit_models > 0:
        py_files = py_files[: args.limit_models]
        print(f"[LIMIT] processing first {len(py_files)} pending models")

    for idx, py_path in enumerate(py_files, 1):
        name = slug(py_path.stem)
        m_dir = out_dir / f"tmp_run_{name}"
        m_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(py_files)}] Processing {name}...")

        try:
            if name not in model_db:
                log_skip(name, "Missing Metadata")
                continue

            model_entry = model_db[name]
            prm = model_entry.get("prm", {})
            transform_val = prm.get("transform")
            target_h = get_resolution_from_transform_file(
                transform_val, transforms_dir, default_h=ds_cfg["default_h"],
            )

            if temp_dl_dir.exists():
                shutil.rmtree(temp_dl_dir)
            temp_dl_dir.mkdir(parents=True, exist_ok=True)
            pth = Path(hf_hub_download(source_repo, f"{py_path.stem}.pth", cache_dir=str(temp_dl_dir)))

            # Sanity-load once here; worker reloads for conversion.
            spec = importlib.util.spec_from_file_location("mod", py_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            model = mod.Net(
                in_shape=(1, 3, target_h, target_h),
                out_shape=(ds_cfg["num_classes"],),
                prm=prm,
                device="cpu",
            )
            ckpt = torch.load(pth, map_location="cpu")
            model.load_state_dict(
                ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt,
                strict=False,
            )
            model.eval()

            convert_ctx = {
                "py_path": str(py_path),
                "pth_path": str(pth),
                "prm": prm,
                "target_h": target_h,
                "num_classes": ds_cfg["num_classes"],
                "mean": list(ds_cfg["mean"]),
                "std": list(ds_cfg["std"]),
                "dataset": args.dataset,
                "imagenet_root": ds_cfg.get("imagenet_root", ""),
            }

            acc_fp = acc_fp16 = acc_mixed = acc_int = acc_dynamic = None
            fp32_p = fp16_p = mixed_p = int8_p = dynamic_p = None
            fp32_ref = None

            if args.quant_mode in ("all", "fp32"):
                fp32_p = m_dir / f"{name}_fp32.tflite"
                acc_fp = run_convert_and_eval(
                    "FP32 Conversion", "FP32", "fp32", fp32_p, data_root, convert_ctx, ds_cfg,
                )
                fp32_ref = fp32_p
                data_fp32[name] = {"accuracy": acc_fp, "transform": transform_val}

            if args.quant_mode in ("all", "fp16"):
                if fp32_ref is None:
                    fp32_ref = m_dir / f"{name}_fp32.tflite"
                    if not (fp32_ref.exists() and fp32_ref.stat().st_size > 0):
                        export_fp32_reference(convert_ctx, fp32_ref, data_root)
                fp16_p = m_dir / f"{name}_fp16.tflite"
                acc_fp16 = run_convert_and_eval(
                    "FP16 Conversion", "FP16", "fp16", fp16_p, data_root, convert_ctx, ds_cfg, fp32_ref,
                )
                data_fp16[name] = {"accuracy": acc_fp16, "transform": transform_val}

            if args.quant_mode in ("all", "mixed"):
                if fp32_ref is None:
                    fp32_ref = m_dir / f"{name}_fp32.tflite"
                    if not (fp32_ref.exists() and fp32_ref.stat().st_size > 0):
                        export_fp32_reference(convert_ctx, fp32_ref, data_root)
                mixed_p = m_dir / f"{name}_mixed.tflite"
                acc_mixed = run_convert_and_eval(
                    "Mixed Precision Conversion", "MIXED", "mixed", mixed_p, data_root, convert_ctx, ds_cfg, fp32_ref,
                )
                data_mixed[name] = {"accuracy": acc_mixed, "transform": transform_val}

            if args.quant_mode in ("all", "static"):
                if fp32_ref is None:
                    fp32_ref = m_dir / f"{name}_fp32.tflite"
                    if not (fp32_ref.exists() and fp32_ref.stat().st_size > 0):
                        export_fp32_reference(convert_ctx, fp32_ref, data_root)
                int8_p = m_dir / f"{name}_int8.tflite"
                acc_int = run_convert_and_eval(
                    "Static INT8 Conversion", "STATIC", "static", int8_p, data_root, convert_ctx, ds_cfg, fp32_ref,
                )
                data_int8[name] = {"accuracy": acc_int, "transform": transform_val}

            if args.quant_mode in ("all", "dynamic"):
                if fp32_ref is None:
                    fp32_ref = m_dir / f"{name}_fp32.tflite"
                    if not (fp32_ref.exists() and fp32_ref.stat().st_size > 0):
                        export_fp32_reference(convert_ctx, fp32_ref, data_root)
                dynamic_p = m_dir / f"{name}_dynamic.tflite"
                acc_dynamic = run_convert_and_eval(
                    "Dynamic INT8 Conversion", "DYNAMIC", "dynamic", dynamic_p, data_root, convert_ctx, ds_cfg, fp32_ref,
                )
                data_dynamic[name] = {"accuracy": acc_dynamic, "transform": transform_val}

            if args.push_hf:
                prefix = ds_cfg["hf_prefix"]
                print(f"   [LOG] Syncing to Hugging Face ({target_repo})...")
                if fp32_p is not None:
                    upload_with_retry(fp32_p, f"fp32/{prefix}/{name}.tflite", target_repo)
                if fp16_p is not None:
                    upload_with_retry(fp16_p, f"fp16/{prefix}/{name}.tflite", target_repo)
                if mixed_p is not None:
                    upload_with_retry(mixed_p, f"mixed/{prefix}/{name}.tflite", target_repo)
                if int8_p is not None:
                    upload_with_retry(int8_p, f"int8/{prefix}/{name}.tflite", target_repo)
                if dynamic_p is not None:
                    upload_with_retry(dynamic_p, f"dynamic/{prefix}/{name}.tflite", target_repo)

                if fp32_p is not None:
                    snap = m_dir / "temp_all_models_fp32.json"
                    with open(snap, "w") as f:
                        json.dump(data_fp32, f, indent=2)
                    upload_with_retry(snap, f"fp32/{prefix}/all_models.json", target_repo)
                if fp16_p is not None:
                    snap = m_dir / "temp_all_models_fp16.json"
                    with open(snap, "w") as f:
                        json.dump(data_fp16, f, indent=2)
                    upload_with_retry(snap, f"fp16/{prefix}/all_models.json", target_repo)
                if mixed_p is not None:
                    snap = m_dir / "temp_all_models_mixed.json"
                    with open(snap, "w") as f:
                        json.dump(data_mixed, f, indent=2)
                    upload_with_retry(snap, f"mixed/{prefix}/all_models.json", target_repo)
                if int8_p is not None:
                    snap = m_dir / "temp_all_models_int8.json"
                    with open(snap, "w") as f:
                        json.dump(data_int8, f, indent=2)
                    upload_with_retry(snap, f"int8/{prefix}/all_models.json", target_repo)
                if dynamic_p is not None:
                    snap = m_dir / "temp_all_models_dynamic.json"
                    with open(snap, "w") as f:
                        json.dump(data_dynamic, f, indent=2)
                    upload_with_retry(snap, f"dynamic/{prefix}/all_models.json", target_repo)

            if fp32_p is not None:
                with open(json_fp32_path, "w") as f:
                    json.dump(data_fp32, f, indent=2)
            if fp16_p is not None:
                with open(json_fp16_path, "w") as f:
                    json.dump(data_fp16, f, indent=2)
            if mixed_p is not None:
                with open(json_mixed_path, "w") as f:
                    json.dump(data_mixed, f, indent=2)
            if int8_p is not None:
                with open(json_int8_path, "w") as f:
                    json.dump(data_int8, f, indent=2)
            if dynamic_p is not None:
                with open(json_dynamic_path, "w") as f:
                    json.dump(data_dynamic, f, indent=2)
            mark_as_done(name, history_file)

            parts = [f"DONE {name}"]
            if acc_fp is not None:
                parts.append(f"FP32: {acc_fp:.4f}")
            if acc_fp16 is not None:
                parts.append(f"FP16: {acc_fp16:.4f}")
            if acc_mixed is not None:
                parts.append(f"MIXED: {acc_mixed:.4f}")
            if acc_int is not None:
                parts.append(f"INT8: {acc_int:.4f}")
            if acc_dynamic is not None:
                parts.append(f"DYNAMIC: {acc_dynamic:.4f}")
            parts.append(f"Transform: {transform_val}")
            print(" | ".join(parts))

        except Exception as e:
            print(f"FAIL {name}: {e}")
            log_fail(name, args.quant_mode, str(e))
        finally:
            shutil.rmtree(m_dir, ignore_errors=True)
            gc.collect()


if __name__ == "__main__":
    main()
