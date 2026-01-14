#!/usr/bin/env python3
"""
run.py — inference profiler that uses existing Train.Train.eval(test_loader) when available.
- CLI: python -m ab.nn.run --model-class ab.nn.nn.ComplexNet.Net --dataset cifar-10 --config img-classification --no-profiler
- Output folder: ab/nn/stat/run/<config>_<architecture>-<timestamp> (or <architecture>-<timestamp>)
- Output filename: windows_devicetype.json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

import torch
from torch import nn

try:
    import psutil
except Exception:
    psutil = None


# ---------------- utilities ----------------

def import_by_path(path: str):
    mod_path, _, cls_name = path.rpartition(".")
    if not mod_path:
        raise ImportError(f"Invalid import path: {path}")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)

def get_device_type() -> str:
    try:
        name = platform.node()
        if name:
            return name
    except Exception:
        pass
    return "unknown_device"

def sanitize_filename(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in s)



def ensure_outdir(path: str) -> str:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    return path


def sanitize_name(name: str) -> str:
    if not name:
        return "run"
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in name)


def extract_arch_name(model_class_path: str) -> str:
    parts = model_class_path.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def default_outpath(model_name: str, config: Optional[str] = None) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    this_dir = os.path.dirname(__file__)  # expected: ab/nn

    if config:
        folder_base = f"{config}_{model_name}"
    else:
        folder_base = model_name or "model"

    folder_safe = sanitize_name(folder_base)
    folder_name = f"{folder_safe}-{ts}"

    run_dir = os.path.join(this_dir, "stat", "run", folder_name)
    os.makedirs(run_dir, exist_ok=True)

    # Anonymized filename
    device = sanitize_filename(get_device_type())
    filename = f"windows_{device}.json"
    return os.path.join(run_dir, filename)


# ---------------- system sampling ----------------

def sample_nvidia_smi():
    if shutil.which("nvidia-smi") is None:
        return None
    q = "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits"
    try:
        out = subprocess.check_output(["nvidia-smi"] + q.split(), stderr=subprocess.DEVNULL)
        out = out.decode("utf-8").strip()
        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                idx, name, driver, total, used, free, util = parts[:7]
                gpus.append({
                    "index": int(idx),
                    "name": name,
                    "driver": driver,
                    "memory_total_mb": int(total),
                    "memory_used_mb": int(used),
                    "memory_free_mb": int(free),
                    "util_percent": (int(util) if util != "N/A" else None)
                })
        return gpus
    except Exception:
        return None


def sample_system():
    if psutil is None:
        return None
    vm = psutil.virtual_memory()
    p = psutil.Process()
    try:
        mem = p.memory_info().rss
    except Exception:
        mem = None
    return {
        "total_ram_bytes": vm.total,
        "available_ram_bytes": vm.available,
        "used_ram_bytes": vm.used,
        "ram_percent": vm.percent,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "process_rss_bytes": mem
    }


# ---------------- dataset + model builders ----------------

def build_dataset_loader(dataset: str, batch_size: int, num_workers: int = 2):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    ds_name = (dataset or "").lower()
    transform = transforms.Compose([transforms.ToTensor()])

    if ds_name == "cifar10":
        ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    elif ds_name == "cifar100":
        ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        num_classes = 100
    elif ds_name.startswith("cifar"):
        ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset '{dataset}' (only cifar10 / cifar100 supported).")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, num_classes


def build_model(model_class: str,
                in_shape: torch.Size,
                out_shape: Tuple[int, ...],
                device: torch.device,
                errors: List[str]) -> nn.Module:
    cls = import_by_path(model_class)
    model = None

    try:
        prm = {"lr": 0.01, "momentum": 0.9}
        model = cls(in_shape=in_shape, out_shape=out_shape, prm=prm, device=device)
        return model
    except TypeError as e:
        errors.append(f"nn_style_ctor_failed: {repr(e)}")
    except Exception as e:
        errors.append(f"nn_style_ctor_other_error: {repr(e)}")

    try:
        model = cls()
        return model
    except TypeError:
        pass
    except Exception as e:
        errors.append(f"default_ctor_failed: {repr(e)}")

    try:
        model = cls(num_classes=out_shape[0])
        return model
    except Exception as e:
        errors.append(f"num_classes_ctor_failed: {repr(e)}")
        raise RuntimeError(f"Failed to construct model for class '{model_class}'. Errors: {errors}")


# ---------------- run_inference ----------------

def run_inference(config: Optional[str],
                  model_class: str,
                  checkpoint: Optional[str],
                  dataset: str,
                  num_batches: int,
                  batch_size: int,
                  outpath: str,
                  use_profiler: bool,
                  debug: bool,
                  force_cpu: bool):
    start_dt = datetime.utcnow()
    start_epoch = time.time()
    errors: List[str] = []

    # architecture name
    model_name = extract_arch_name(model_class)

    # device
    device = torch.device("cpu")
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")

    # dataset loader
    test_loader, num_classes = build_dataset_loader(dataset, batch_size)

    # infer shapes by peeking a sample (for potential model build fallback)
    try:
        sample_inputs, _ = next(iter(test_loader))
    except StopIteration:
        raise RuntimeError("Test loader is empty - cannot run inference.")
    in_shape = sample_inputs.shape
    out_shape = (num_classes,)

    # attempt to build a model (best-effort; Train.Train may construct its own model)
    model = None
    try:
        model = build_model(model_class, in_shape, out_shape, device, errors)
    except Exception:
        model = None

    # prepare containers
    timeline: List[dict[str, Any]] = []
    op_totals: dict[str, dict[str, Any]] = {}
    prof_traces: List[str] = []
    peak_cpu_rss = 0
    peak_gpu_mb = 0.0
    batches_run = 0
    metric_result = None
    eval_used = False

    # coarse "before eval" snapshot
    timeline.append({
        "phase": "before_eval",
        "ts": datetime.utcnow().isoformat() + "Z",
        "sys": sample_system(),
        "gpus": sample_nvidia_smi()
    })

    try:
        train_mod = importlib.import_module("ab.nn.util.Train")
        TrainClass = getattr(train_mod, "Train", None)
        if TrainClass is not None:
            # try best-effort instantiation
            trainer = None
            try:
                sig = inspect.signature(TrainClass.__init__)
                ctor_kwargs = {}
                # if Train accepts 'model' and we built one, pass it
                if "model" in sig.parameters and model is not None:
                    ctor_kwargs["model"] = model
                if "device" in sig.parameters:
                    ctor_kwargs["device"] = device
                try:
                    trainer = TrainClass(**ctor_kwargs)
                except Exception:
                    # fallback to no-arg constructor
                    trainer = TrainClass()
            except Exception:
                try:
                    trainer = TrainClass()
                except Exception:
                    trainer = None

            if trainer is not None and hasattr(trainer, "eval"):
                try:
                    # call trainer.eval(test_loader) per provided signature
                    metric_result = trainer.eval(test_loader)
                    eval_used = True
                except Exception as e:
                    errors.append(f"trainer_eval_exception: {repr(e)}")
        else:
            # try module-level eval(test_loader)
            eval_fn = getattr(train_mod, "eval", None)
            if callable(eval_fn):
                try:
                    metric_result = eval_fn(test_loader)
                    eval_used = True
                except Exception as e:
                    errors.append(f"module_eval_exception: {repr(e)}")
    except Exception as e:
        errors.append(f"train_import_failed: {repr(e)}")

    # coarse "after eval" snapshot
    timeline.append({
        "phase": "after_eval",
        "ts": datetime.utcnow().isoformat() + "Z",
        "sys": sample_system(),
        "gpus": sample_nvidia_smi()
    })


    if not eval_used:
        loader_iter = iter(test_loader)
        with torch.no_grad():
            for bi in range(num_batches):
                try:
                    inputs, labels = next(loader_iter)
                except StopIteration:
                    break

                # --- before_batch snapshot (per-batch, only in fallback) ---
                timeline.append({
                    "phase": "before_batch",
                    "batch": bi,
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "sys": sample_system(),
                    "gpus": sample_nvidia_smi()
                })

                inputs, labels = inputs.to(device), labels.to(device)
                if "ComplexNet" in model_class:
                    if not torch.is_complex(inputs):
                        inputs = inputs.to(torch.complex64)

                if use_profiler and hasattr(torch, "profiler"):
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device.type == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    try:
                        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                            outputs = model(inputs) if model is not None else None
                        ka = prof.key_averages()
                        for op in ka:
                            name = op.key
                            cpu_us = getattr(op, "cpu_time_total", None)
                            if cpu_us is None:
                                cpu_us = getattr(op, "self_cpu_time_total", 0.0)
                            cpu_ms = float(cpu_us) / 1000.0 if cpu_us is not None else 0.0
                            rec = op_totals.setdefault(name, {"cpu_ms": 0.0, "calls": 0, "mem_bytes": 0})
                            rec["cpu_ms"] += cpu_ms
                            rec["calls"] += int(getattr(op, "count", 1))
                        trace_file = os.path.join(os.path.dirname(outpath), f"trace_batch{bi}.json")
                        try:
                            prof.export_chrome_trace(trace_file)
                            prof_traces.append(trace_file)
                        except Exception:
                            pass
                    except Exception as e:
                        errors.append(f"profiler_batch_failed_{bi}: {repr(e)}")
                        if model is not None:
                            outputs = model(inputs)
                else:
                    if model is not None:
                        outputs = model(inputs)
                    else:
                        outputs = None

                # --- after_batch snapshot ---
                timeline.append({
                    "phase": "after_batch",
                    "batch": bi,
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "sys": sample_system(),
                    "gpus": sample_nvidia_smi()
                })

                # update peaks
                last_sys = timeline[-1]["sys"] if timeline and "sys" in timeline[-1] else None
                if last_sys and last_sys.get("process_rss_bytes") is not None:
                    peak_cpu_rss = max(peak_cpu_rss, last_sys["process_rss_bytes"])

                if device.type == "cuda":
                    try:
                        torch.cuda.synchronize()
                        d = torch.cuda.current_device()
                        mem_mb = torch.cuda.memory_allocated(d) / (1024 ** 2)
                        peak_gpu_mb = max(peak_gpu_mb, mem_mb)
                    except Exception:
                        pass

                batches_run += 1

    else:
        # estimate batches_run from timeline coarse info if metric_result doesn't include it
        batches_run = None

    # summarize ops
    ops = [{"name": k, "cpu_ms": v["cpu_ms"], "calls": v["calls"], "mem_bytes": v.get("mem_bytes", None)} for k, v in op_totals.items()]
    ops_sorted = sorted(ops, key=lambda x: x["cpu_ms"], reverse=True)[:40]

    end_dt = datetime.utcnow()
    duration_seconds = (end_dt - start_dt).total_seconds()
    duration_ms = int(duration_seconds * 1000)

    vm = psutil.virtual_memory() if psutil else None
    total_ram_kb = f"{vm.total // 1024} kB" if vm else None
    free_ram_kb = f"{vm.free // 1024} kB" if (vm and hasattr(vm, "free")) else None
    available_ram_kb = f"{vm.available // 1024} kB" if vm else None
    cached_kb = f"{getattr(vm, 'cached', 0) // 1024} kB" if (vm and hasattr(vm, "cached")) else None

    cpu_cores = psutil.cpu_count(logical=True) if psutil else None
    processors = [{"vendor_id": platform.processor() or None, "model": platform.machine() or None}]

    device_type = get_device_type()
    os_version = platform.platform()

    final_report = {
        "model_name": model_name,
        "device_type": device_type,
        "os_version": os_version,
        "valid": len(errors) == 0,
        "emulator": False,
        "error_message": errors[0] if errors else None,
        "duration_ms": duration_ms,
        "duration_seconds": duration_seconds,
        "device_analytics": {
            "timestamp": start_epoch,
            "memory_info": {
                "total_ram_kb": total_ram_kb,
                "free_ram_kb": free_ram_kb,
                "available_ram_kb": available_ram_kb,
                "cached_kb": cached_kb,
            },
            "cpu_info": {
                "cpu_cores": cpu_cores,
                "processors": processors,
                "arm_architecture": None
            },
            "timeline": timeline,
            "profile": {
                "top_ops": ops_sorted,
                "profiler_traces": prof_traces,
                "peak_cpu_rss_bytes": peak_cpu_rss,
                "peak_gpu_mb": peak_gpu_mb,
                "metric_result": metric_result
            }
        }
    }

    ensure_outdir(outpath)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, default=str)
    print(f"[inference_profiler] saved: {outpath}")
    return outpath


# ---------------- CLI ----------------

def main():
    p = argparse.ArgumentParser(prog="ab.nn.run", description="Profile inference and save JSON report.")
    p.add_argument("--config", type=str, default=None, help="Optional config name (stored in JSON only).")
    p.add_argument("--model-class", type=str, required=True, help="Model class path, e.g. ab.nn.nn.ComplexNet.Net")
    p.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path to load.")
    p.add_argument("--dataset", type=str, default="cifar10", help="Dataset name (cifar10 or cifar100).")
    p.add_argument("--num-batches", type=int, default=10, help="Number of batches to run.")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    p.add_argument("--out", dest="outpath", type=str, default=None, help="Optional output JSON path.")
    p.add_argument("--no-profiler", dest="no_profiler", action="store_true", help="Disable torch.profiler even if available.")
    p.add_argument("--debug", action="store_true", help="Enable debug prints.")
    p.add_argument("--force-cpu", action="store_true", help="Force using CPU even if CUDA available.")
    args = p.parse_args()

    if args.outpath is None:
        model_name = extract_arch_name(args.model_class)
        outpath = default_outpath(model_name, config=args.config)
    else:
        outpath = args.outpath

    run_inference(config=args.config,
                  model_class=args.model_class,
                  checkpoint=args.checkpoint,
                  dataset=args.dataset,
                  num_batches=args.num_batches,
                  batch_size=args.batch_size,
                  outpath=outpath,
                  use_profiler=(not args.no_profiler),
                  debug=args.debug,
                  force_cpu=args.force_cpu)


if __name__ == "__main__":
    main()
