#!/usr/bin/env python3
import sys
import os
import argparse
import json
import re
import subprocess
import importlib.util
import shutil
import time
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

for p in [out_dir, data_root, temp_dl_dir]:
    p.mkdir(parents=True, exist_ok=True)

# --- 2. IMPORTS ---
import torch
import torchvision
import torchvision.transforms as T
import ai_edge_torch
import tensorflow as tf
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

def mark_as_done(name: str):
    current_list = []
    if HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0:
        try:
            with open(HISTORY_FILE, "r") as f: current_list = json.load(f)
        except: current_list = []
    
    if name not in current_list:
        current_list.append(name)
        with open(HISTORY_FILE, "w") as f:
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

def eval_tflite_acc_subprocess(tflite_path: Path, data_root: Path, batch_size: int = 100) -> Dict[str, Any]:
    payload = {"tflite_path": str(tflite_path), "data_root": str(data_root), "batch_size": int(batch_size), "limit": 1000}
    code = r"""
import json, sys, os, numpy as np, tensorflow as tf, torchvision as tv, torch
p = json.loads(sys.stdin.read())
try:
    interp = tf.lite.Interpreter(model_path=p["tflite_path"]); interp.allocate_tensors()
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
    print(json.dumps({"ok": True, "acc": float(correct)/total}))
except Exception as e: print(json.dumps({"ok": False, "error": str(e)}))
"""
    proc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True)
    out, _ = proc.communicate(input=json.dumps(payload))
    try: return json.loads(out.strip().splitlines()[-1])
    except: return {"ok": False}

# ------------------------
# Main
# ------------------------
def main():
    arch_dir = dataset_root / "ab" / "nn" / "nn"
    transforms_dir = dataset_root / "ab" / "nn" / "transform"
    local_models_json = dataset_root / "all_models.json"

    ap = argparse.ArgumentParser()
    ap.add_argument("--push-hf", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--hf-token", default=DEFAULT_HF_TOKEN)
    args = ap.parse_args()
    if args.hf_token: os.environ["HF_TOKEN"] = args.hf_token

    if args.push_hf: create_repo(TARGET_REPO, repo_type="model", exist_ok=True)

    try:
        downloaded_json = hf_hub_download(repo_id=SOURCE_REPO, filename="all_models.json", local_dir=str(out_dir), force_download=True)
        shutil.copy(downloaded_json, local_models_json)
    except: pass

    with open(local_models_json) as f: model_db = json.load(f)

    json_fp32_path = out_dir / "all_models_accuracy_fp32.json"
    json_int8_path = out_dir / "all_models_accuracy_int8.json"
    data_fp32 = load_json_safe(json_fp32_path)
    data_int8 = load_json_safe(json_int8_path)

    processed_history = set(load_json_safe(HISTORY_FILE))
    processed_history.update(data_fp32.keys())

    hf_files = list_repo_files(SOURCE_REPO)
    py_files = sorted([p for p in arch_dir.rglob("*.py") if f"{p.stem}.pth" in hf_files])

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
            transform_val = prm.get("transform", "default")
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
            
            # --- FP32 ---
            fp32_p = m_dir / f"{name}_fp32.tflite"
            ai_edge_torch.convert(model, dummy_input).export(str(fp32_p))
            res_fp = eval_tflite_acc_subprocess(fp32_p, data_root)
            acc_fp = res_fp.get("acc", 0.0)
            
            # --- INT8 ---
            int8_p = m_dir / f"{name}_int8.tflite"
            def rep():
                tfm = T.Compose([
                    T.ToTensor(), 
                    T.Resize((target_h, target_h), antialias=True),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
                d = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=tfm)
                for j in range(50): yield [d[j][0].unsqueeze(0).numpy()]
            
            ai_edge_torch.convert(model, dummy_input, 
                _ai_edge_converter_flags={'optimizations': [tf.lite.Optimize.DEFAULT], 'representative_dataset': rep, 
                'target_spec': {'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]}, 
                'inference_input_type': tf.int8, 'inference_output_type': tf.int8}).export(str(int8_p))
            res_int = eval_tflite_acc_subprocess(int8_p, data_root)
            acc_int = res_int.get("acc", 0.0)

            # --- DATA PREPARATION ---
            data_fp32[name] = {"accuracy": acc_fp, "transform": transform_val}
            data_int8[name] = {"accuracy": acc_int, "transform": transform_val}

            # --- STRUCTURED UPLOAD ---
            if args.push_hf:
                prefix = "img-classification_cifar-10_acc"
                print(f"   [LOG] Syncing to Hugging Face...")
                upload_with_retry(fp32_p, f"fp32/{prefix}/{name}.tflite", TARGET_REPO)
                upload_with_retry(int8_p, f"int8/{prefix}/{name}.tflite", TARGET_REPO)
                
                # Snapshot logic for Cloud JSONs
                snap_fp32_path = m_dir / "temp_all_models_fp32.json"
                snap_int8_path = m_dir / "temp_all_models_int8.json"
                with open(snap_fp32_path, "w") as f: json.dump(data_fp32, f, indent=2)
                with open(snap_int8_path, "w") as f: json.dump(data_int8, f, indent=2)
                
                upload_with_retry(snap_fp32_path, f"fp32/{prefix}/all_models.json", TARGET_REPO)
                upload_with_retry(snap_int8_path, f"int8/{prefix}/all_models.json", TARGET_REPO)

            # --- UPDATE MASTER LOCALLY ---
            with open(json_fp32_path, "w") as f: json.dump(data_fp32, f, indent=2)
            with open(json_int8_path, "w") as f: json.dump(data_int8, f, indent=2)
            mark_as_done(name)
            
            print(f"DONE {name} | FP32: {acc_fp:.4f} | INT8: {acc_int:.4f} | Transform: {transform_val}")

        except Exception as e:
            print(f"FAIL {name}: {e}")
        finally:
            shutil.rmtree(m_dir, ignore_errors=True)

if __name__ == "__main__": main()
