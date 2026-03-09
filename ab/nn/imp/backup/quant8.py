#!/usr/bin/env python3
"""
torch2tflite-all.py

MODIFICATIONS:
- "android_devices" JSON structure compliant with user request.
- ALL TIME VALUES converted to NANOSECONDS (ns).
- ALL SIZE VALUES converted to KILOBYTES (KB).
- RAM info fetched from /proc/meminfo (already in KB).
- STRICTLY NO FILE PATHS SAVED IN JSON.
- REMOVED: host_binary, arch_path, weights path, tflite_path.
- REMOVED: workstation (torch version), eval_backend.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import ai_edge_torch  # type: ignore
import tensorflow as tf  # type: ignore

HF_OK = True
try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:
    HF_OK = False


# ------------------------
# Repo auto-discovery
# ------------------------
def find_repo_root(start: Path, repo_name: str) -> Optional[Path]:
    cur = start.resolve()
    for _ in range(12):
        if cur.name == repo_name:
            return cur
        cur = cur.parent
    return None


def find_nn_dataset_root(lite_root: Path) -> Path:
    for k in ("NN_DATASET_ROOT", "NN_DATASET_DIR", "NN_DATASET_PATH"):
        v = os.environ.get(k)
        if v:
            p = Path(v).expanduser().resolve()
            if (p / "ab" / "nn" / "nn").exists():
                return p

    candidates = [
        lite_root.parent / "nn-dataset",
        lite_root.parent.parent / "nn-dataset",
        Path.home() / "nn-dataset",
    ]

    cur = Path.cwd().resolve()
    for _ in range(12):
        candidates.append(cur / "nn-dataset")
        cur = cur.parent

    for c in candidates:
        if (c / "ab" / "nn" / "nn").exists():
            return c.resolve()

    raise RuntimeError(
        "Could not locate nn-dataset repo.\n"
        "Fix (one-time): set env var, e.g.\n"
        "  export NN_DATASET_ROOT=/path/to/nn-dataset\n"
        "Then rerun:\n"
        "  python torch2tflite-all.py\n"
    )


def autodiscover_defaults() -> Tuple[Path, Path, Path, str]:
    script_dir = Path(__file__).resolve().parent
    lite_root = find_repo_root(script_dir, "nn-lite")
    if lite_root is None:
        lite_root = script_dir

    dataset_root = find_nn_dataset_root(lite_root)

    arch_dir = dataset_root / "ab" / "nn" / "nn"
    out_dir = lite_root / "ab" / "nn" / "stat" / "imp"
    data_root = lite_root / "ab" / "lite" / "data"
    hf_repo = os.environ.get("NN_HF_REPO", "NN-Dataset/checkpoints-epoch-50")
    return arch_dir.resolve(), out_dir.resolve(), data_root.resolve(), hf_repo


# ------------------------
# Utils
# ------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s[:200]


def fsize_kb(p: Path) -> float:
    """Returns file size in KB."""
    if p.exists():
        return p.stat().st_size / 1024.0
    return 0.0


def now() -> float:
    return time.time()


def run_cmd(cmd: List[str], check: bool = False) -> str:
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=check)
    return cp.stdout or ""


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def load_existing_stats(stats_path: Path) -> Optional[Dict[str, Any]]:
    if not stats_path.exists():
        return None
    try:
        with open(stats_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def which(cmd: str) -> Optional[str]:
    try:
        out = run_cmd(["bash", "-lc", f"command -v {cmd}"])
        p = out.strip()
        return p if p else None
    except Exception:
        return None


# ------------------------
# CIFAR-10
# ------------------------
def get_cifar10(data_root: Path, batch_size: int, num_workers: int):
    tfm = T.Compose([T.ToTensor()])
    train = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=tfm)
    test = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=tfm)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


# ------------------------
# TFLite INT8 accuracy (CRASH-SAFE via subprocess)
# ------------------------
def _tflite_eval_worker(tflite_path: str, data_root: str, batch_size: int, max_eval_batches: int) -> Dict[str, Any]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

    import numpy as _np
    import tensorflow as _tf
    import torchvision as _tv
    import torchvision.transforms as _T
    import torch as _torch

    tfm = _T.Compose([_T.ToTensor()])
    test = _tv.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=tfm)
    loader = _torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)

    backend = "tf.lite.Interpreter"
    Interpreter = _tf.lite.Interpreter

    try:
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
    except Exception as e:
        return {"ok": False, "backend": backend, "error": f"interpreter_failed: {e}"}

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_index = in_det["index"]
    out_index = out_det["index"]

    in_dtype = in_det["dtype"]
    in_scale, in_zp = in_det.get("quantization", (0.0, 0))
    if in_scale is None or in_scale == 0:
        in_scale, in_zp = 1.0, 0

    in_shape = in_det["shape"]
    expects_nhwc = (
        len(in_shape) == 4
        and int(in_shape[1]) == 32
        and int(in_shape[2]) == 32
        and int(in_shape[3]) == 3
    )

    correct, total = 0, 0
    for bi, (x, y) in enumerate(loader):
        if max_eval_batches and bi >= max_eval_batches:
            break

        x_np = x.numpy().astype(_np.float32)  # NCHW
        if expects_nhwc:
            x_np = _np.transpose(x_np, (0, 2, 3, 1))  # NHWC

        if in_dtype == _np.int8:
            q = _np.round(x_np / in_scale + in_zp).astype(_np.int32)
            q = _np.clip(q, -128, 127).astype(_np.int8)
        elif in_dtype == _np.uint8:
            q = _np.round(x_np / in_scale + in_zp).astype(_np.int32)
            q = _np.clip(q, 0, 255).astype(_np.uint8)
        else:
            q = x_np.astype(in_dtype)

        bs = q.shape[0]
        for i in range(bs):
            interpreter.set_tensor(in_index, q[i : i + 1])
            interpreter.invoke()
            out = interpreter.get_tensor(out_index)
            pred = int(_np.argmax(out, axis=-1).reshape(-1)[0])
            true = int(y[i].item())
            correct += (pred == true)
            total += 1

    acc = float(correct) / float(total) if total else 0.0
    return {"ok": True, "backend": backend, "acc": acc}


def eval_tflite_int8_acc_subprocess(
    tflite_path: Path,
    data_root: Path,
    batch_size: int,
    max_eval_batches: int,
    timeout_sec: int = 900,
) -> Dict[str, Any]:
    payload = {
        "tflite_path": str(tflite_path),
        "data_root": str(data_root),
        "batch_size": int(batch_size),
        "max_eval_batches": int(max_eval_batches),
    }

    code = r"""
import json, sys, os
def _tflite_eval_worker(tflite_path: str, data_root: str, batch_size: int, max_eval_batches: int):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import numpy as _np
    import tensorflow as _tf
    import torchvision as _tv
    import torchvision.transforms as _T
    import torch as _torch
    tfm = _T.Compose([_T.ToTensor()])
    test = _tv.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=tfm)
    loader = _torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    backend = "tf.lite.Interpreter"
    try:
        interpreter = _tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
    except Exception as e:
        return {"ok": False, "backend": backend, "error": f"interpreter_failed: {e}"}
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_index = in_det["index"]
    out_index = out_det["index"]
    in_dtype = in_det["dtype"]
    in_scale, in_zp = in_det.get("quantization", (0.0, 0))
    if in_scale is None or in_scale == 0:
        in_scale, in_zp = 1.0, 0
    in_shape = in_det["shape"]
    expects_nhwc = (len(in_shape)==4 and int(in_shape[1])==32 and int(in_shape[2])==32 and int(in_shape[3])==3)
    correct=0; total=0
    for bi,(x,y) in enumerate(loader):
        if max_eval_batches and bi>=max_eval_batches:
            break
        x_np = x.numpy().astype(_np.float32)
        if expects_nhwc:
            x_np = _np.transpose(x_np,(0,2,3,1))
        if in_dtype == _np.int8:
            q = _np.round(x_np / in_scale + in_zp).astype(_np.int32)
            q = _np.clip(q,-128,127).astype(_np.int8)
        elif in_dtype == _np.uint8:
            q = _np.round(x_np / in_scale + in_zp).astype(_np.int32)
            q = _np.clip(q,0,255).astype(_np.uint8)
        else:
            q = x_np.astype(in_dtype)
        bs=q.shape[0]
        for i in range(bs):
            interpreter.set_tensor(in_index, q[i:i+1])
            interpreter.invoke()
            out = interpreter.get_tensor(out_index)
            pred = int(_np.argmax(out,axis=-1).reshape(-1)[0])
            true = int(y[i].item())
            correct += (pred==true)
            total += 1
    acc = float(correct)/float(total) if total else 0.0
    return {"ok": True, "backend": backend, "acc": acc}

p = json.loads(sys.stdin.read())
res = _tflite_eval_worker(p["tflite_path"], p["data_root"], p["batch_size"], p["max_eval_batches"])
print(json.dumps(res))
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        out, _ = proc.communicate(input=json.dumps(payload), timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"ok": False, "error": "tflite_eval_timeout", "rc": -1, "raw": ""}

    rc = proc.returncode
    if rc != 0:
        return {"ok": False, "error": f"tflite_eval_crashed_rc={rc}", "rc": rc, "raw": out[-4000:]}

    try:
        line = out.strip().splitlines()[-1].strip()
        res = json.loads(line)
        res["rc"] = rc
        return res
    except Exception:
        return {"ok": False, "error": "tflite_eval_bad_output", "rc": rc, "raw": out[-4000:]}


# ------------------------
# Architecture loading (your dataset interface)
# ------------------------
def import_py(py_path: Path) -> Any:
    uniq = py_path.stem + "_" + hashlib.md5(str(py_path).encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(uniq, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Import failed for {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[uniq] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def get_supported_hparams(mod: Any) -> List[str]:
    if hasattr(mod, "supported_hyperparameters") and callable(mod.supported_hyperparameters):
        hp = mod.supported_hyperparameters()
        if isinstance(hp, (set, list, tuple)):
            return list(hp)
        if isinstance(hp, dict):
            return list(hp.keys())
    return ["lr", "momentum"]


def default_prm(keys: List[str]) -> Dict[str, Any]:
    base = {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0, "dropout": 0.0, "epochs": 1}
    return {k: base.get(k, 0.0) for k in keys}


def build_net(mod: Any, device: torch.device) -> nn.Module:
    if not hasattr(mod, "Net"):
        raise RuntimeError("No Net class found")
    Net = getattr(mod, "Net")
    if not isinstance(Net, type) or not issubclass(Net, nn.Module):
        raise RuntimeError("Net exists but is not nn.Module")

    in_shape = (1, 3, 32, 32)
    out_shape = (10,)  # CIFAR-10
    prm = default_prm(get_supported_hparams(mod))
    return Net(in_shape=in_shape, out_shape=out_shape, prm=prm, device=device)


def load_pth(model: nn.Module, pth_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(str(pth_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise RuntimeError("Unsupported checkpoint format")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    return {"loaded": True, "missing_keys": list(missing), "unexpected_keys": list(unexpected)}


# ------------------------
# HuggingFace
# ------------------------
def hf_list_files(repo_id: str) -> List[str]:
    if not HF_OK:
        return []
    return list_repo_files(repo_id=repo_id)


def hf_download(repo_id: str, filename: str, cache_dir: Path) -> Path:
    if not HF_OK:
        raise RuntimeError("huggingface_hub not installed")
    p = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir))
    return Path(p)


# ------------------------
# Fast weights index once
# ------------------------
def build_weights_stem_index(arch_dir: Path, hf_files: List[str]) -> Tuple[set, Dict[str, Path], set]:
    stems: set = set()
    local_map: Dict[str, Path] = {}

    for p in arch_dir.rglob("*.pth"):
        stems.add(p.stem)
        local_map.setdefault(p.stem, p)

    hf_stems: set = set()
    for f in hf_files:
        if f.endswith(".pth"):
            st = Path(f).stem
            stems.add(st)
            hf_stems.add(st)

    return stems, local_map, hf_stems


# ------------------------
# INT8 conversion (STATIC)
# ------------------------
def representative_dataset(train_loader, num_batches: int) -> Callable[[], Iterable[List[np.ndarray]]]:
    def gen():
        n = 0
        for x, _ in train_loader:
            yield [x.numpy().astype(np.float32)]
            n += 1
            if n >= num_batches:
                break
    return gen


def convert_int8(model: nn.Module, rep_fn, out_path: Path) -> None:
    sample = (torch.randn(1, 3, 32, 32),)

    flags = {
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "representative_dataset": rep_fn,
        "target_spec": {"supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]},
        "inference_input_type": tf.int8,
        "inference_output_type": tf.int8,
    }

    edge_model = ai_edge_torch.convert(model.eval(), sample, _ai_edge_converter_flags=flags)
    safe_mkdir(out_path.parent)
    edge_model.export(str(out_path))


# ------------------------
# Android benchmarking (benchmark_model) + AUTO-SETUP
# ------------------------
@dataclass
class Bench:
    backend: str
    iterations: int
    avg: Optional[float]
    min: Optional[float]
    max: Optional[float]
    std: Optional[float]
    time_unit: Optional[str]
    raw: str


def adb(args: List[str]) -> str:
    return run_cmd(["adb"] + args, check=False)


def adb_shell(cmd: str) -> str:
    return adb(["shell", cmd])


def adb_devices() -> List[str]:
    out = run_cmd(["adb", "devices"])
    online = []
    for line in out.splitlines():
        if "\tdevice" in line:
            online.append(line.split()[0])
    return online


def ensure_benchmark_on_device(script_dir: Path) -> Dict[str, Any]:
    if which("adb") is None:
        return {"ok": False, "reason": "adb_not_found"}

    devs = adb_devices()
    if len(devs) == 0:
        return {"ok": False, "reason": "no_device"}
    chosen = devs[0]

    abi = run_cmd(["adb", "shell", "getprop", "ro.product.cpu.abi"]).strip()
    if "arm64" in abi:
        host_bin = script_dir / "android_tools" / "benchmark_model_arm64"
    elif "armeabi" in abi:
        host_bin = script_dir / "android_tools" / "benchmark_model_armeabi_v7a"
    else:
        return {"ok": False, "reason": f"unsupported_abi:{abi}"}

    if not host_bin.exists():
        return {"ok": False, "reason": f"missing_host_binary:{host_bin}"}

    chk = run_cmd(["adb", "shell", "ls", "/data/local/tmp/benchmark_model", "2>/dev/null || echo MISSING"]).strip()
    if "MISSING" in chk or "No such file" in chk:
        run_cmd(["adb", "push", str(host_bin), "/data/local/tmp/benchmark_model"], check=True)
        run_cmd(["adb", "shell", "chmod", "+x", "/data/local/tmp/benchmark_model"], check=True)

    # REMOVED "host_binary" from the return dictionary
    return {"ok": True, "device": chosen, "abi": abi}


def adb_getprop(key: str) -> str:
    return adb_shell(f"getprop {key}").strip()


def get_android_device_identifier() -> Dict[str, str]:
    device_type = adb_getprop("ro.product.model") or "Unknown"
    # prefer display id; fallback to build fingerprint or android release
    os_version = (
        adb_getprop("ro.build.display.id")
        or adb_getprop("ro.build.fingerprint")
        or adb_getprop("ro.build.version.release")
        or "Unknown"
    )
    device_id = f"android - {device_type} - {os_version}"
    return {"device_id": device_id, "device_type": device_type, "os_version": os_version}


def get_android_memory_info() -> Dict[str, int]:
    """Parses /proc/meminfo from Android device. Values in KB."""
    out = adb_shell("cat /proc/meminfo")
    info = {}
    for line in out.splitlines():
        # Line fmt: "MemTotal:        5864560 kB"
        parts = line.split(":")
        if len(parts) == 2:
            key = parts[0].strip()
            # The second part often looks like " 5864560 kB"
            val_str = parts[1].strip().split()[0]
            if val_str.isdigit():
                info[key] = int(val_str)
    
    return {
        "total_ram_kb": info.get("MemTotal", 0),
        "free_ram_kb": info.get("MemFree", 0),
        "available_ram_kb": info.get("MemAvailable", 0),
        "cached_kb": info.get("Cached", 0)
    }


def to_ns(val: Optional[float], unit: Optional[str]) -> Optional[float]:
    """Converts a time value to nanoseconds based on unit."""
    if val is None:
        return None
    if not unit or unit == "unknown":
        return val # Fallback, assume raw
    
    u = unit.lower()
    if u == "ns":
        return val
    elif u == "us":
        return val * 1000.0
    elif u == "ms":
        return val * 1_000_000.0
    elif u == "s":
        return val * 1_000_000_000.0
    return val


def parse_bench(out: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], int, Optional[str], str]:
    """
    Tries to parse a line like:
      ... count=50 ... avg=1234us min=1200us max=1300us std=50us
    Returns raw parsed values and the detected unit.
    """
    lines = [l.strip() for l in out.splitlines() if ("count=" in l and "avg=" in l and "min=" in l and "max=" in l)]
    if not lines:
        return None, None, None, None, 0, None, ""

    last = lines[-1]

    def grab_num_and_unit(k: str) -> Tuple[Optional[float], Optional[str]]:
        m = re.search(rf"\b{k}=([0-9]+(?:\.[0-9]+)?)([a-zA-Z]+)?\b", last)
        if not m:
            return None, None
        val = float(m.group(1))
        unit = m.group(2) if m.group(2) else None
        return val, unit

    avg, u_avg = grab_num_and_unit("avg")
    mn, u_min = grab_num_and_unit("min")
    mx, u_max = grab_num_and_unit("max")
    std, u_std = grab_num_and_unit("std")

    it = 0
    mc = re.search(r"\bcount=([0-9]+)\b", last)
    if mc:
        it = int(mc.group(1))

    # pick the first non-empty unit among fields
    unit = u_avg or u_min or u_max or u_std

    return avg, mn, mx, std, it, unit, last


def android_run_bench_one(model_on_device: str, backend: str, runs: int, warmup: int, threads: int) -> Bench:
    bench_bin = "/data/local/tmp/benchmark_model"

    if backend == "cpu":
        flags = "--use_xnnpack=false"
    elif backend == "gpu":
        flags = "--use_gpu"
    elif backend == "nnapi":
        flags = "--use_nnapi"
    else:
        flags = ""

    cmd = (
        f"chmod +x {bench_bin} && "
        f"{bench_bin} --graph={model_on_device} "
        f"--num_threads={threads} --warmup_runs={warmup} --num_runs={runs} {flags}"
    )
    out = adb_shell(cmd)
    
    # Parse
    avg, mn, mx, std, it, unit, parsed_line = parse_bench(out)

    # CONVERT TO NANOSECONDS
    avg_ns = to_ns(avg, unit)
    min_ns = to_ns(mn, unit)
    max_ns = to_ns(mx, unit)
    std_ns = to_ns(std, unit)
    
    # Store unit as 'ns' if conversion happened, else whatever we got
    final_unit = "ns" if (unit and unit != "unknown") else (unit or "unknown")

    return Bench(
        backend=backend,
        iterations=(it or runs),
        avg=avg_ns,
        min=min_ns,
        max=max_ns,
        std=std_ns,
        time_unit=final_unit,
        raw=out,
    )


def bench_on_android(model_local: Path, model_name: str, runs: int, warmup: int, threads: int) -> Dict[str, Any]:
    dev_dir = "/data/local/tmp/nnlite_models"
    dev_model = f"{dev_dir}/{model_name}.tflite"
    adb_shell(f"mkdir -p {dev_dir}")
    adb(["push", str(model_local), dev_model])

    # Run benchmarks
    cpu = android_run_bench_one(dev_model, "cpu", runs, warmup, threads)
    gpu = android_run_bench_one(dev_model, "gpu", runs, warmup, threads)
    nnapi = android_run_bench_one(dev_model, "nnapi", runs, warmup, threads)

    # Calculate best backend (using avg duration in ns)
    # Filter out None/failed runs
    candidates = {}
    if cpu.avg is not None: candidates["cpu"] = cpu.avg
    if gpu.avg is not None: candidates["gpu"] = gpu.avg
    if nnapi.avg is not None: candidates["npu"] = nnapi.avg  # Mapped to NPU for consistency

    if candidates:
        best_backend_key = min(candidates, key=candidates.get)
        best_duration = candidates[best_backend_key]
    else:
        best_backend_key = "unknown"
        best_duration = 0.0

    # Gather memory info (in KB)
    mem_info = get_android_memory_info()

    # Construct the flat structure requested
    payload = {
        "model_name": model_name,
        # "device_type" and "os_version" will be injected by the caller
        "valid": True,
        "emulator": False,
        "iterations": runs,
        "duration": best_duration,   # In NS
        "unit": best_backend_key,    # "cpu", "gpu", or "npu"
        
        # CPU (Values in NS)
        "cpu_duration": cpu.avg,
        "cpu_min_duration": cpu.min,
        "cpu_max_duration": cpu.max,
        "cpu_std_dev": cpu.std,
        
        # GPU (Values in NS)
        "gpu_duration": gpu.avg,
        "gpu_min_duration": gpu.min,
        "gpu_max_duration": gpu.max,
        "gpu_std_dev": gpu.std,
        
        # NPU (mapped from NNAPI, Values in NS)
        "npu_duration": nnapi.avg,
        "npu_min_duration": nnapi.min,
        "npu_max_duration": nnapi.max,
        "npu_std_dev": nnapi.std,
        
        # RAM (Values in KB)
        "total_ram_kb": mem_info.get("total_ram_kb", 0),
        "free_ram_kb": mem_info.get("free_ram_kb", 0),
        "available_ram_kb": mem_info.get("available_ram_kb", 0),
        "cached_kb": mem_info.get("cached_kb", 0),

        # Fixed Input Dims (CIFAR-10 standard)
        "in_dim_0": 1,
        "in_dim_1": 3,
        "in_dim_2": 32,
        "in_dim_3": 32,
        
        "device_analytics": {} 
    }
    
    return payload


# ------------------------
# Resume logic (no fp32 dependency)
# ------------------------
def stage_ok(stages: Dict[str, Any], key: str) -> bool:
    return stages.get(key) == "ok"


def is_completed(existing: Dict[str, Any], tflite_path: Path, android_requested: bool) -> bool:
    stages = existing.get("stages", {})
    if not stage_ok(stages, "int8_convert"):
        return False
    if stages.get("int8_eval") not in ("ok", "error"):
        return False
    if not tflite_path.exists():
        return False
    if android_requested and stages.get("android_bench") not in ("ok", "error"):
        return False
    return True


# ------------------------
# main
# ------------------------
def main():
    arch_def, out_def, data_def, hf_def = autodiscover_defaults()

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch-dir", default=str(arch_def))
    ap.add_argument("--out-dir", default=str(out_def))
    ap.add_argument("--hf-repo", default=hf_def, help="HF repo for weights")
    ap.add_argument("--data-root", default=str(data_def))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--rep-batches", type=int, default=50)
    ap.add_argument("--max-models", type=int, default=0)
    ap.add_argument("--max-eval-batches", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--redo-android", action="store_true", default=False)
    ap.add_argument("--android-runs", type=int, default=20)
    ap.add_argument("--android-warmup", type=int, default=5)
    ap.add_argument("--android-threads", type=int, default=4)
    ap.add_argument("--tflite-eval-timeout", type=int, default=900)
    args = ap.parse_args()

    arch_dir = Path(args.arch_dir)
    out_dir = Path(args.out_dir)
    data_root = Path(args.data_root)
    safe_mkdir(out_dir)
    cache_dir = out_dir / "_hf_cache"
    safe_mkdir(cache_dir)

    device = torch.device("cpu")

    train_loader, _ = get_cifar10(data_root, args.batch_size, args.num_workers)

    hf_files = hf_list_files(args.hf_repo) if args.hf_repo else []
    stems, local_map, hf_stems = build_weights_stem_index(arch_dir, hf_files)

    all_py = sorted([p for p in arch_dir.rglob("*.py") if p.is_file()])
    py_files = [p for p in all_py if p.stem in stems]  # only-with-weights always

    script_dir = Path(__file__).resolve().parent
    android_setup = ensure_benchmark_on_device(script_dir)
    android_possible = bool(android_setup.get("ok", False))

    print(
        f"[INFO] arch_dir={arch_dir}\n"
        f"[INFO] out_dir={out_dir}\n"
        f"[INFO] data_root={data_root}\n"
        f"[INFO] hf_repo={args.hf_repo}\n"
        f"[INFO] total_py={len(all_py)} | weighted_py={len(py_files)} | resume={args.resume}\n"
        f"[ANDROID] setup={android_setup} | enabled={android_possible}\n"
    )

    if args.max_models and args.max_models > 0:
        py_files = py_files[: args.max_models]

    ok = fail = skip = 0

    for py_path in py_files:
        name = slug(py_path.stem)
        out_model_dir = out_dir / name
        safe_mkdir(out_model_dir)

        stats_path = out_model_dir / "stats.json"
        tflite_path = out_model_dir / f"{name}.int8.tflite"

        existing = load_existing_stats(stats_path) if args.resume else None
        if existing and is_completed(existing, tflite_path, android_requested=android_possible) and not args.redo_android:
            print(f"[RESUME-SKIP] {name} (already completed)")
            skip += 1
            continue

        stats: Dict[str, Any] = existing if existing else {
            "model_name": name,
            # "arch_path": str(py_path), # DELETED: Path not stored
            "created_at": now(),
            # "workstation": {"torch": torch.__version__}, # DELETED: workstation info removed
            "weights": {},
            "android_setup": android_setup,
            "stages": {},
            "errors": [],
        }

        stats["updated_at"] = now()
        stats.setdefault("stages", {})
        stats.setdefault("errors", [])
        stats["android_setup"] = android_setup

        # IMPORTANT: remove legacy "android" key if present (prof said remove previous statistics)
        if "android" in stats:
            del stats["android"]

        atomic_write_json(stats_path, stats)

        try:
            pth_name = f"{py_path.stem}.pth"
            ckpt_path: Optional[Path] = None

            if py_path.stem in local_map:
                ckpt_path = local_map[py_path.stem]
                # DELETED "path": str(ckpt_path) from dictionary
                stats["weights"] = {"available": True, "source": "local"}
            elif args.hf_repo and (py_path.stem in hf_stems):
                ckpt_path = hf_download(args.hf_repo, pth_name, cache_dir)
                # DELETED "path": str(ckpt_path) from dictionary
                stats["weights"] = {"available": True, "source": "hf"}
            else:
                stats["weights"] = {"available": False, "expected": pth_name}

            if not stats["weights"].get("available", False):
                stats["stages"]["skipped"] = "weights_not_found"
                atomic_write_json(stats_path, stats)
                print(f"[SKIP] {name} (weights not found)")
                skip += 1
                continue

            if stats["stages"].get("import_build_load") != "ok":
                stats["stages"]["import_build_load"] = "running"
                atomic_write_json(stats_path, stats)

                mod = import_py(py_path)
                model = build_net(mod, device=device).to(device).eval()

                ld = load_pth(model, ckpt_path)  # type: ignore
                stats["weights"].update(ld)

                stats["stages"]["import_build_load"] = "ok"
                atomic_write_json(stats_path, stats)
            else:
                mod = import_py(py_path)
                model = build_net(mod, device=device).to(device).eval()
                # CHANGED: Use ckpt_path variable, not stats["weights"]["path"]
                if ckpt_path:
                    load_pth(model, ckpt_path)
                else:
                    # Fallback just in case, though logic suggests we shouldn't reach here without it
                    raise RuntimeError("Resume failed: weight path not resolved.")

            if stats["stages"].get("int8_convert") != "ok":
                stats["stages"]["int8_convert"] = "running"
                atomic_write_json(stats_path, stats)

                rep_fn = representative_dataset(train_loader, args.rep_batches)
                convert_int8(model.cpu(), rep_fn, tflite_path)

                stats["int8"] = {
                    # "tflite_path": str(tflite_path), # DELETED
                    "tflite_size_kb": fsize_kb(tflite_path), # CHANGED TO KB
                }

                stats["stages"]["int8_convert"] = "ok"
                atomic_write_json(stats_path, stats)

            if stats["stages"].get("int8_eval") not in ("ok", "error"):
                stats["stages"]["int8_eval"] = "running"
                atomic_write_json(stats_path, stats)

                res = eval_tflite_int8_acc_subprocess(
                    tflite_path=tflite_path,
                    data_root=data_root,
                    batch_size=args.batch_size,
                    max_eval_batches=args.max_eval_batches,
                    timeout_sec=args.tflite_eval_timeout,
                )

                stats.setdefault("int8", {})
                # stats["int8"]["eval_backend"] = res.get("backend") # DELETED: eval_backend removed
                if res.get("ok"):
                    stats["int8"]["top1_acc"] = float(res["acc"])
                    stats["stages"]["int8_eval"] = "ok"
                else:
                    stats["int8"]["eval_error"] = res.get("error")
                    stats["int8"]["eval_rc"] = res.get("rc")
                    stats["int8"]["eval_raw_tail"] = res.get("raw")
                    stats["stages"]["int8_eval"] = "error"

                atomic_write_json(stats_path, stats)

            # ANDROID BENCH: MODIFIED STRUCTURE
            if android_possible:
                if args.redo_android or stats["stages"].get("android_bench") not in ("ok", "error"):
                    stats["stages"]["android_bench"] = "running"
                    atomic_write_json(stats_path, stats)

                    try:
                        dev = get_android_device_identifier()
                        device_id = dev["device_id"]

                        android_payload = bench_on_android(
                            model_local=tflite_path,
                            model_name=name,
                            runs=args.android_runs,
                            warmup=args.android_warmup,
                            threads=args.android_threads,
                        )
                        # Inject device identifiers from the device we actually ran on
                        android_payload["device_type"] = dev["device_type"]
                        android_payload["os_version"] = dev["os_version"]

                        stats.setdefault("android_devices", {})
                        stats["android_devices"][device_id] = android_payload

                        # remove any legacy android flat key if present
                        if "android" in stats:
                            del stats["android"]

                        stats["stages"]["android_bench"] = "ok"
                    except Exception as e:
                        stats.setdefault("android_devices", {})
                        stats["android_devices"]["error"] = {"error": str(e)}
                        stats["stages"]["android_bench"] = "error"

                    atomic_write_json(stats_path, stats)

            ok += 1
            int8v = stats.get("int8", {}).get("top1_acc", None)
            print(f"[OK] {name} | int8_acc={int8v} | tflite_kb={fsize_kb(tflite_path):.2f} | android={android_possible}")

        except Exception as e:
            fail += 1
            stats["errors"].append(str(e))
            for k, v in list(stats.get("stages", {}).items()):
                if v == "running":
                    stats["stages"][k] = "error"
            atomic_write_json(stats_path, stats)
            print(f"[FAIL] {name}: {e}")

    print(f"\nDone. ok={ok}, fail={fail}, skip={skip}, out={out_dir}")


if __name__ == "__main__":
    main()
