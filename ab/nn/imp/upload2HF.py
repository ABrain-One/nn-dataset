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
SOURCE_REPO = "NN-Dataset/checkpoints-epoch-50"
DEFAULT_HF_TOKEN = "" # Set via --hf-token or HF_TOKEN env var or paste here
# ---------------------

# --- 1. SETUP PATHS ---
script_path = Path(__file__).resolve()
dataset_root = script_path.parents[3]

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

HISTORY_FILES = {
    "all": HISTORY_FILE,
    "fp32": out_dir / "upload_history_fp32.json",
    "fp16": out_dir / "upload_history_fp16.json",
    "mixed": out_dir / "upload_history_mixed.json",
    "static": out_dir / "upload_history_static.json",
    "dynamic": out_dir / "upload_history_dynamic.json",
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

def get_resolution_from_transform_file(transform_name: str, transforms_dir: Path) -> int:
    if not transform_name: 
        print(f"   [DEBUG] No transform name. Defaulting to 32.")
        return 32 
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
            except: pass
    print(f"   [DEBUG] Transform file '{transform_name}' NOT FOUND. Defaulting to 32.")
    return 32

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

def eval_tflite_acc_subprocess(tflite_path: Path, data_root: Path, batch_size: int = 100) -> Dict[str, Any]:
    payload = {"tflite_path": str(tflite_path), "data_root": str(data_root), "batch_size": int(batch_size), "limit": 1000}
    code = r"""
import json, sys, os, numpy as np, tensorflow as tf, torchvision as tv, torch
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
    h, w = (spatial[-2], spatial[-1]) if len(spatial) >= 2 else (32, 32)
    tfm = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize((h, w), antialias=True),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
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
        "acc": float(correct) / total,
        "in_dtype": str(in_dtype),
        "out_dtype": str(out_det["dtype"]),
    }))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
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

def rep_dataset(target_h: int, data_root: Path):
    def rep():
        tfm = T.Compose([
            T.ToTensor(),
            T.Resize((target_h, target_h), antialias=True),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        d = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=tfm)
        for j in range(50):
            yield [d[j][0].unsqueeze(0).numpy()]
    return rep

def file_size_kib(path):
    return Path(path).stat().st_size / 1024

def verify_tflite(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"TFLite export missing or empty: {path}")

def print_conversion_summary(label, out_path, elapsed_ms, acc, original_path=None, io_types=None):
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
    print(f"   CIFAR-10 accuracy:      {acc:.4f}")
    print(f"   Total time:             {elapsed_ms:.2f} ms")
    print(f"   -> Successfully converted {label} model.")

def run_convert_and_eval(process_label, save_label, convert_fn, out_path, data_root, original_path=None):
    print(f"   [PROCESS] {process_label}...")
    t0 = time.perf_counter()
    convert_fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    verify_tflite(out_path)
    res = eval_tflite_acc_subprocess(out_path, data_root)
    if not res.get("ok"):
        raise RuntimeError(f"Accuracy eval failed: {res.get('error', 'unknown error')}")
    acc = res.get("acc", 0.0)
    print_conversion_summary(
        save_label, out_path, elapsed_ms, acc, original_path,
        io_types={"in_dtype": res.get("in_dtype"), "out_dtype": res.get("out_dtype")},
    )
    return acc

def mixed_precision_converter_flags(target_h: int, data_root: Path) -> dict:
    """Calibrated PTQ: quantize supported ops to INT8; unsupported ops (e.g. PADV2) stay float."""
    import tensorflow as tf
    return {
        'optimizations': [tf.lite.Optimize.DEFAULT],
        'representative_dataset': rep_dataset(target_h, data_root),
        'target_spec': {
            'supported_ops': [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ],
        },
    }

def export_fp32_reference(model, dummy_input, ref_path):
    import ai_edge_torch
    print(f"   [BASELINE] FP32 reference not found, exporting for size comparison...")
    t0 = time.perf_counter()
    ai_edge_torch.convert(model, dummy_input).export(str(ref_path))
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
    ap.add_argument("--push-hf", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--hf-token", default=DEFAULT_HF_TOKEN)
    ap.add_argument("--format", choices=["tflite", "onnx"], default="tflite")
    args = ap.parse_args()
    if args.hf_token: os.environ["HF_TOKEN"] = args.hf_token

    # ------------------------
    # ROUTE TO ONNX PIPELINE
    # ------------------------
    if args.format == "onnx":
        from onnx_pipeline import run_onnx_pipeline
        run_onnx_pipeline(args, dataset_root)
        return

    # --- IMPORT TFLITE ONLY WHEN NEEDED ---
    import ai_edge_torch
    import tensorflow as tf

    if args.push_hf: create_repo(TARGET_REPO, repo_type="model", exist_ok=True)

    try:
        downloaded_json = hf_hub_download(repo_id=SOURCE_REPO, filename="all_models.json", local_dir=str(out_dir), force_download=True)
        shutil.copy(downloaded_json, local_models_json)
    except: pass

    with open(local_models_json) as f: model_db = json.load(f)

    json_fp32_path = out_dir / "all_models_accuracy_fp32.json"
    json_fp16_path = out_dir / "all_models_accuracy_fp16.json"
    json_mixed_path = out_dir / "all_models_accuracy_mixed.json"
    json_int8_path = out_dir / "all_models_accuracy_int8.json"
    json_dynamic_path = out_dir / "all_models_accuracy_dynamic.json"
    data_fp32 = load_json_safe(json_fp32_path)
    data_fp16 = load_json_safe(json_fp16_path)
    data_mixed = load_json_safe(json_mixed_path)
    data_int8 = load_json_safe(json_int8_path)
    data_dynamic = load_json_safe(json_dynamic_path)

    history_file = HISTORY_FILES[args.quant_mode]
    processed_history = set(load_json_safe(history_file))
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

    hf_files = list_repo_files(SOURCE_REPO)
    py_files = sorted([p for p in arch_dir.rglob("*.py") if f"{p.stem}.pth" in hf_files])

    print(f"\n[{args.quant_mode.upper()} RUN] Models to process: {len(py_files)}")

    for idx, py_path in enumerate(py_files, 1):
        name = slug(py_path.stem)
        if args.resume and name in processed_history: continue
            
        m_dir = out_dir / f"tmp_run_{name}"
        m_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(py_files)}] Processing {name}...")
        
        try:
            if name not in model_db:
                log_skip(name, "Missing Metadata")
                continue

            model_entry = model_db[name]
            prm = model_entry.get("prm", {})
            
            # --- EXTRACT TRANSFORM STRING ---
            transform_val = prm.get("transform")
            target_h = get_resolution_from_transform_file(transform_val, transforms_dir)

            if temp_dl_dir.exists(): shutil.rmtree(temp_dl_dir)
            temp_dl_dir.mkdir(parents=True, exist_ok=True)
            pth = Path(hf_hub_download(SOURCE_REPO, f"{py_path.stem}.pth", cache_dir=str(temp_dl_dir)))
            
            spec = importlib.util.spec_from_file_location("mod", py_path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            model = mod.Net(in_shape=(1,3,32,32), out_shape=(10,), prm=prm, device="cpu")
            ckpt = torch.load(pth, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt, strict=False)
            model.eval()
            dummy_input = (torch.randn(1, 3, target_h, target_h),)

            acc_fp = acc_fp16 = acc_mixed = acc_int = acc_dynamic = None
            fp32_p = fp16_p = mixed_p = int8_p = dynamic_p = None
            fp32_ref = None

            # --- FP32 ---
            if args.quant_mode in ("all", "fp32"):
                fp32_p = m_dir / f"{name}_fp32.tflite"
                acc_fp = run_convert_and_eval(
                    "FP32 Conversion",
                    "FP32",
                    lambda: ai_edge_torch.convert(model, dummy_input).export(str(fp32_p)),
                    fp32_p,
                    data_root,
                )
                fp32_ref = fp32_p
                data_fp32[name] = {"accuracy": acc_fp, "transform": transform_val}

            # --- FP16 (post-training) ---
            if args.quant_mode in ("all", "fp16"):
                if fp32_ref is None:
                    fp32_ref = m_dir / f"{name}_fp32.tflite"
                    if not (fp32_ref.exists() and fp32_ref.stat().st_size > 0):
                        export_fp32_reference(model, dummy_input, fp32_ref)
                fp16_p = m_dir / f"{name}_fp16.tflite"
                acc_fp16 = run_convert_and_eval(
                    "FP16 Conversion",
                    "FP16",
                    lambda: ai_edge_torch.convert(
                        model, dummy_input,
                        _ai_edge_converter_flags={
                            'optimizations': [tf.lite.Optimize.DEFAULT],
                            'target_spec': {'supported_types': [tf.float16]},
                        },
                    ).export(str(fp16_p)),
                    fp16_p,
                    data_root,
                    fp32_ref,
                )
                data_fp16[name] = {"accuracy": acc_fp16, "transform": transform_val}

            # --- MIXED PRECISION (calibrated INT8 + float fallback ops) ---
            if args.quant_mode in ("all", "mixed"):
                if fp32_ref is None:
                    fp32_ref = m_dir / f"{name}_fp32.tflite"
                    if not (fp32_ref.exists() and fp32_ref.stat().st_size > 0):
                        export_fp32_reference(model, dummy_input, fp32_ref)
                mixed_p = m_dir / f"{name}_mixed.tflite"
                acc_mixed = run_convert_and_eval(
                    "Mixed Precision Conversion",
                    "MIXED",
                    lambda: ai_edge_torch.convert(
                        model, dummy_input,
                        _ai_edge_converter_flags=mixed_precision_converter_flags(target_h, data_root),
                    ).export(str(mixed_p)),
                    mixed_p,
                    data_root,
                    fp32_ref,
                )
                data_mixed[name] = {"accuracy": acc_mixed, "transform": transform_val}

            # --- STATIC INT8 ---
            if args.quant_mode in ("all", "static"):
                if fp32_ref is None:
                    fp32_ref = m_dir / f"{name}_fp32.tflite"
                    if not (fp32_ref.exists() and fp32_ref.stat().st_size > 0):
                        export_fp32_reference(model, dummy_input, fp32_ref)
                int8_p = m_dir / f"{name}_int8.tflite"
                acc_int = run_convert_and_eval(
                    "Static INT8 Conversion",
                    "STATIC",
                    lambda: ai_edge_torch.convert(
                        model, dummy_input,
                        _ai_edge_converter_flags={
                            'optimizations': [tf.lite.Optimize.DEFAULT],
                            'representative_dataset': rep_dataset(target_h, data_root),
                            'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]},
                            'inference_input_type': tf.int8,
                            'inference_output_type': tf.int8,
                        },
                    ).export(str(int8_p)),
                    int8_p,
                    data_root,
                    fp32_ref,
                )
                data_int8[name] = {"accuracy": acc_int, "transform": transform_val}

            # --- DYNAMIC INT8 ---
            if args.quant_mode in ("all", "dynamic"):
                if fp32_ref is None:
                    fp32_ref = m_dir / f"{name}_fp32.tflite"
                    if not (fp32_ref.exists() and fp32_ref.stat().st_size > 0):
                        export_fp32_reference(model, dummy_input, fp32_ref)
                dynamic_p = m_dir / f"{name}_dynamic.tflite"
                acc_dynamic = run_convert_and_eval(
                    "Dynamic INT8 Conversion",
                    "DYNAMIC",
                    lambda: ai_edge_torch.convert(
                        model, dummy_input,
                        _ai_edge_converter_flags={'optimizations': [tf.lite.Optimize.DEFAULT]},
                    ).export(str(dynamic_p)),
                    dynamic_p,
                    data_root,
                    fp32_ref,
                )
                data_dynamic[name] = {"accuracy": acc_dynamic, "transform": transform_val}

            # --- STRUCTURED UPLOAD ---
            if args.push_hf:
                prefix = "img-classification_cifar-10_acc"
                print(f"   [LOG] Syncing to Hugging Face...")
                if fp32_p is not None:
                    upload_with_retry(fp32_p, f"fp32/{prefix}/{name}.tflite", TARGET_REPO)
                if fp16_p is not None:
                    upload_with_retry(fp16_p, f"fp16/{prefix}/{name}.tflite", TARGET_REPO)
                if mixed_p is not None:
                    upload_with_retry(mixed_p, f"mixed/{prefix}/{name}.tflite", TARGET_REPO)
                if int8_p is not None:
                    upload_with_retry(int8_p, f"int8/{prefix}/{name}.tflite", TARGET_REPO)
                if dynamic_p is not None:
                    upload_with_retry(dynamic_p, f"dynamic/{prefix}/{name}.tflite", TARGET_REPO)

                # Snapshot logic for Cloud JSONs
                if fp32_p is not None:
                    snap_fp32_path = m_dir / "temp_all_models_fp32.json"
                    with open(snap_fp32_path, "w") as f: json.dump(data_fp32, f, indent=2)
                    upload_with_retry(snap_fp32_path, f"fp32/{prefix}/all_models.json", TARGET_REPO)
                if fp16_p is not None:
                    snap_fp16_path = m_dir / "temp_all_models_fp16.json"
                    with open(snap_fp16_path, "w") as f: json.dump(data_fp16, f, indent=2)
                    upload_with_retry(snap_fp16_path, f"fp16/{prefix}/all_models.json", TARGET_REPO)
                if mixed_p is not None:
                    snap_mixed_path = m_dir / "temp_all_models_mixed.json"
                    with open(snap_mixed_path, "w") as f: json.dump(data_mixed, f, indent=2)
                    upload_with_retry(snap_mixed_path, f"mixed/{prefix}/all_models.json", TARGET_REPO)
                if int8_p is not None:
                    snap_int8_path = m_dir / "temp_all_models_int8.json"
                    with open(snap_int8_path, "w") as f: json.dump(data_int8, f, indent=2)
                    upload_with_retry(snap_int8_path, f"int8/{prefix}/all_models.json", TARGET_REPO)
                if dynamic_p is not None:
                    snap_dynamic_path = m_dir / "temp_all_models_dynamic.json"
                    with open(snap_dynamic_path, "w") as f: json.dump(data_dynamic, f, indent=2)
                    upload_with_retry(snap_dynamic_path, f"dynamic/{prefix}/all_models.json", TARGET_REPO)

            # --- UPDATE MASTER LOCALLY ---
            if fp32_p is not None:
                with open(json_fp32_path, "w") as f: json.dump(data_fp32, f, indent=2)
            if fp16_p is not None:
                with open(json_fp16_path, "w") as f: json.dump(data_fp16, f, indent=2)
            if mixed_p is not None:
                with open(json_mixed_path, "w") as f: json.dump(data_mixed, f, indent=2)
            if int8_p is not None:
                with open(json_int8_path, "w") as f: json.dump(data_int8, f, indent=2)
            if dynamic_p is not None:
                with open(json_dynamic_path, "w") as f: json.dump(data_dynamic, f, indent=2)
            mark_as_done(name, history_file)

            parts = [f"DONE {name}"]
            if acc_fp is not None: parts.append(f"FP32: {acc_fp:.4f}")
            if acc_fp16 is not None: parts.append(f"FP16: {acc_fp16:.4f}")
            if acc_mixed is not None: parts.append(f"MIXED: {acc_mixed:.4f}")
            if acc_int is not None: parts.append(f"INT8: {acc_int:.4f}")
            if acc_dynamic is not None: parts.append(f"DYNAMIC: {acc_dynamic:.4f}")
            parts.append(f"Transform: {transform_val}")
            print(" | ".join(parts))

        except Exception as e:
            print(f"FAIL {name}: {e}")
            log_fail(name, args.quant_mode, str(e))
        finally:
            shutil.rmtree(m_dir, ignore_errors=True)
            gc.collect()

if __name__ == "__main__": main()