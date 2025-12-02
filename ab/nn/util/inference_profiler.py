#!/usr/bin/env python3
"""
Inference Profiler for nn-dataset Models
------------------------------------------------

This script runs any supported nn-dataset model in inference (eval) mode,
collecting the following information into a structured JSON report:

  • Accuracy metric
  • System analytics (RAM, CPU, OS, device)
  • Batch-wise timeline snapshots
  • Optional torch.profiler traces (top ops, chrome trace files)

Generalized Model Support:
- Automatically inspects constructors and adapts shapes
- Works for models such as: ComplexNet, ResNet, AirNet, etc.

The report format follows the structure provided by the professor:
{
  "model_name": ...,
  "device_type": ...,
  "os_version": ...,
  ...
}

Example (CPU only, no profiler):
  python -m ab.nn.util.inference_profiler model-class ab.nn.nn.ComplexNet.Net --dataset cifar10 --batch-size 8 --num-batches 2 --out results/infer/complexnet-prof.json --no-profiler

Example (with profiler + trace files):
  python -m ab.nn.util.inference_profiler --model-class ab.nn.nn.ResNet.Net --dataset cifar100 --batch-size 32 --num-batches 5 --out results/infer/resnet-prof.json
"""


from __future__ import annotations

import argparse
import importlib
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




def import_by_path(path: str):
    """
    Import a class by full dotted path:
        package.module.ClassName
    Example: ab.nn.nn.ComplexNet.Net
    """
    mod_path, _, cls_name = path.rpartition(".")
    if not mod_path:
        raise ImportError(f"Invalid import path: {path}")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def ensure_outdir(path: str) -> str:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    return path


def sample_nvidia_smi():
    """Return list of GPU dicts or None if nvidia-smi not available."""
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
    """Return basic system stats or None if psutil not installed."""
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


# ----------------- dataset + model builders -----------------


def build_dataset_loader(dataset: str, batch_size: int, num_workers: int = 2):
    """
    Returns: (DataLoader, num_classes)
      - cifar10  -> 10 classes
      - cifar100 -> 100 classes
    """
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
        # default to cifar10 if user wrote cifar-10, CIFAR_10 etc.
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
    """
    Build model from class path.

    Strategy:
      1) Try NN-dataset convention:
           Net(in_shape=in_shape, out_shape=out_shape, prm=..., device=device)
      2) If that fails, fall back to:
           Net()
           Net(num_classes=out_shape[0])
    """
    cls = import_by_path(model_class)
    model = None

    # 1) NN-dataset style ctor
    try:
        prm = {"lr": 0.01, "momentum": 0.9}
        model = cls(in_shape=in_shape, out_shape=out_shape, prm=prm, device=device)
        return model
    except TypeError as e:
        errors.append(f"nn_style_ctor_failed: {repr(e)}")
    except Exception as e:
        errors.append(f"nn_style_ctor_other_error: {repr(e)}")

    # 2a) plain ctor
    try:
        model = cls()
        return model
    except TypeError:
        pass
    except Exception as e:
        errors.append(f"default_ctor_failed: {repr(e)}")

    # 2b) num_classes ctor
    try:
        model = cls(num_classes=out_shape[0])
        return model
    except Exception as e:
        errors.append(f"num_classes_ctor_failed: {repr(e)}")
        raise RuntimeError(
            f"Failed to construct model for class '{model_class}'. "
            f"Errors: {errors}"
        )



class SimpleAccuracy:

    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, labels):
        preds = outputs.argmax(dim=1)
        self.correct += int((preds == labels).sum().item())
        self.total += labels.size(0)

    def result(self):
        return {"accuracy": (self.correct / self.total) if self.total > 0 else None}




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

    # device
    device = torch.device("cpu")
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")

    # dataset
    test_loader, num_classes = build_dataset_loader(dataset, batch_size)

    # infer shape from real batch
    try:
        sample_inputs, _ = next(iter(test_loader))
    except StopIteration:
        raise RuntimeError("Test loader is empty - cannot run inference.")
    in_shape = sample_inputs.shape  # e.g. [B, 3, 32, 32]
    out_shape = (num_classes,)

    # model
    model = build_model(model_class, in_shape, out_shape, device, errors)
    model_name = getattr(model, "__class__", type(model)).__name__


    if checkpoint:
        try:
            ckpt = torch.load(checkpoint, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            try:
                model.load_state_dict(sd)
            except Exception:
                new_sd = {}
                for k, v in sd.items():
                    new_key = k.replace("module.", "") if k.startswith("module.") else k
                    new_sd[new_key] = v
                model.load_state_dict(new_sd, strict=False)
        except Exception as e:
            errors.append(f"checkpoint_load_failed: {repr(e)}")

    # metric
    metric = SimpleAccuracy()
    metric.reset()

    # state for timeline + profiling
    timeline: List[dict[str, Any]] = []
    op_totals: dict[str, dict[str, Any]] = {}
    prof_traces: List[str] = []
    peak_cpu_rss = 0
    peak_gpu_mb = 0.0
    batches_run = 0

    # iterator (we will manually stop after num_batches)
    loader_iter = iter(test_loader)

    with torch.no_grad():
        for bi in range(num_batches):
            try:
                inputs, labels = next(loader_iter)
            except StopIteration:
                break

            # --- before batch snapshot ---
            ts_before = datetime.utcnow().isoformat() + "Z"
            sys_before = sample_system()
            gpus_before = sample_nvidia_smi()
            timeline.append({
                "phase": "before_batch",
                "batch": bi,
                "ts": ts_before,
                "sys": sys_before,
                "gpus": gpus_before
            })

            # move to device
            inputs, labels = inputs.to(device), labels.to(device)

            # ComplexNet expects complex input
            if "ComplexNet" in model_class:
                if not torch.is_complex(inputs):
                    inputs = inputs.to(torch.complex64)

            # --- run model (with or without profiler) ---
            if use_profiler and hasattr(torch, "profiler"):
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)

                try:
                    with torch.profiler.profile(
                        activities=activities,
                        record_shapes=True,
                        profile_memory=True
                    ) as prof:
                        outputs = model(inputs)

                    # aggregate op stats
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
                        try:
                            mem_b = getattr(op, "self_cpu_memory_usage", 0) or 0
                            rec["mem_bytes"] += int(mem_b)
                        except Exception:
                            pass

                    # export chrome trace
                    try:
                        trace_file = f"{outpath}_trace_batch{bi}.json"
                        prof.export_chrome_trace(trace_file)
                        prof_traces.append(trace_file)
                    except Exception:
                        pass
                except Exception as e:
                    errors.append(f"profiler_batch_failed_{bi}: {repr(e)}")
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            # metric update (prof-style eval logic)
            try:
                metric(outputs, labels)
            except Exception as e:
                errors.append(f"metric_failed_batch_{bi}: {repr(e)}")

            # --- after batch snapshot ---
            ts_after = datetime.utcnow().isoformat() + "Z"
            sys_after = sample_system()
            gpus_after = sample_nvidia_smi()
            timeline.append({
                "phase": "after_batch",
                "batch": bi,
                "ts": ts_after,
                "sys": sys_after,
                "gpus": gpus_after
            })

            # peak memory
            if sys_after and "process_rss_bytes" in sys_after and sys_after["process_rss_bytes"] is not None:
                peak_cpu_rss = max(peak_cpu_rss, sys_after["process_rss_bytes"])

            if device.type == "cuda":
                try:
                    torch.cuda.synchronize()
                    d = torch.cuda.current_device()
                    mem_mb = torch.cuda.memory_allocated(d) / (1024 ** 2)
                    peak_gpu_mb = max(peak_gpu_mb, mem_mb)
                except Exception:
                    pass

            batches_run += 1
            if debug:
                print(f"[inference_profiler] finished batch {bi}", flush=True)

    # metric result
    try:
        metric_result = metric.result()
    except Exception:
        metric_result = None

    # summarize ops
    ops = [
        {
            "name": k,
            "cpu_ms": v["cpu_ms"],
            "calls": v["calls"],
            "mem_bytes": v.get("mem_bytes", None)
        }
        for k, v in op_totals.items()
    ]
    ops_sorted = sorted(ops, key=lambda x: x["cpu_ms"], reverse=True)[:40]

    end_dt = datetime.utcnow()
    duration_seconds = (end_dt - start_dt).total_seconds()
    duration_ms = int(duration_seconds * 1000)

    # system info at end
    vm = psutil.virtual_memory() if psutil else None
    total_ram_kb = f"{vm.total // 1024} kB" if vm else None
    free_ram_kb = f"{vm.free // 1024} kB" if (vm and hasattr(vm, "free")) else None
    available_ram_kb = f"{vm.available // 1024} kB" if vm else None
    cached_kb = f"{getattr(vm, 'cached', 0) // 1024} kB" if (vm and hasattr(vm, "cached")) else None

    cpu_cores = psutil.cpu_count(logical=True) if psutil else None
    processors = [{
        "vendor_id": platform.processor() or None,
        "model": platform.machine() or None,
    }]

    device_type = platform.node() or "unknown_device"
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
            "run_details": {
                "config": config,
                "model_class": model_class,
                "dataset": dataset,
                "device": str(device),
                "num_batches_requested": num_batches,
                "batches_run": batches_run,
                "batch_size": batch_size,
                "metric_result": metric_result,
            },
            "timeline": timeline,
            "profile": {
                "top_ops": ops_sorted,
                "profiler_traces": prof_traces,
                "peak_cpu_rss_bytes": peak_cpu_rss,
                "peak_gpu_mb": peak_gpu_mb
            }
        }
    }

    ensure_outdir(outpath)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, default=str)
    print(f"[inference_profiler] saved: {outpath}")
    return outpath


# ----------------- CLI -----------------


def main():
    p = argparse.ArgumentParser(
        prog="ab.nn.util.inference_profiler",
        description="Profile inference and save JSON report (metrics + hardware + op stats)."
    )
    p.add_argument("--config", type=str, default=None,
                   help="Optional config name (stored in JSON only).")
    p.add_argument("--model-class", type=str, required=True,
                   help="Model class path, e.g. ab.nn.nn.ComplexNet.Net")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional checkpoint path to load.")
    p.add_argument("--dataset", type=str, default="cifar10",
                   help="Dataset name (cifar10 or cifar100).")
    p.add_argument("--num-batches", type=int, default=10,
                   help="Number of batches to run.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size.")
    p.add_argument("--out", dest="outpath", type=str, default="results/infer/profile-report.json",
                   help="Output JSON path.")
    p.add_argument("--no-profiler", dest="no_profiler", action="store_true",
                   help="Disable torch.profiler even if available.")
    p.add_argument("--debug", action="store_true",
                   help="Enable debug prints.")
    p.add_argument("--force-cpu", action="store_true",
                   help="Force using CPU even if CUDA available.")
    args = p.parse_args()

    run_inference(
        config=args.config,
        model_class=args.model_class,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        outpath=args.outpath,
        use_profiler=(not args.no_profiler),
        debug=args.debug,
        force_cpu=args.force_cpu,
    )


if __name__ == "__main__":
    main()
