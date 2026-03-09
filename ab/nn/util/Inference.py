from __future__ import annotations

import importlib
import inspect
import json
import time
import platform
from datetime import datetime
from typing import Any, List, Optional, Tuple

import torch
from torch import nn
import psutil

from ab.nn.util.Util import (
    import_by_path,
    extract_arch_name,
    ensure_outdir,
    get_device_type,
    sample_system,
    sample_nvidia_smi
)


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

    model_name = extract_arch_name(model_class)

    device = torch.device("cpu")
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")

    test_loader, num_classes = build_dataset_loader(dataset, batch_size)

    sample_inputs, _ = next(iter(test_loader))
    in_shape = sample_inputs.shape
    out_shape = (num_classes,)
    in_dim_0 = 1
    in_dim_1 = int(in_shape[2])
    in_dim_2 = int(in_shape[3])
    in_dim_3 = int(in_shape[1])
    model = None
    try:
        model = build_model(model_class, in_shape, out_shape, device, errors)
    except Exception:
        model = None

    timeline: List[dict[str, Any]] = []
    op_totals: dict[str, dict[str, Any]] = {}
    prof_traces: List[str] = []
    peak_cpu_rss = 0
    peak_gpu_mb = 0.0
    batches_run = 0
    metric_result = None
    eval_used = False

    timeline.append({
        "phase": "before_eval",
        "ts": datetime.utcnow().isoformat() + "Z",
        "sys": sample_system(),
        "gpus": sample_nvidia_smi()
    })

    # (existing Train.eval logic unchanged)
    try:
        train_mod = importlib.import_module("ab.nn.util.Train")
        TrainClass = getattr(train_mod, "Train", None)
        if TrainClass is not None:
            trainer = None
            try:
                sig = inspect.signature(TrainClass.__init__)
                ctor_kwargs = {}
                if "model" in sig.parameters and model is not None:
                    ctor_kwargs["model"] = model
                if "device" in sig.parameters:
                    ctor_kwargs["device"] = device
                try:
                    trainer = TrainClass(**ctor_kwargs)
                except Exception:
                    trainer = TrainClass()
            except Exception:
                try:
                    trainer = TrainClass()
                except Exception:
                    trainer = None

            if trainer is not None and hasattr(trainer, "eval"):
                try:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        gpu_start_evt = torch.cuda.Event(enable_timing=True)
                        gpu_end_evt = torch.cuda.Event(enable_timing=True)
                        gpu_start_evt.record()
                    metric_result = trainer.eval(test_loader)
                    if device.type == "cuda":
                        gpu_end_evt.record()
                        torch.cuda.synchronize()
                        gpu_duration_ns = int(gpu_start_evt.elapsed_time(gpu_end_evt) * 1_000_000)
                    else:
                        gpu_duration_ns = None
                    eval_used = True
                except Exception as e:
                    errors.append(f"trainer_eval_exception: {repr(e)}")
    except Exception:
        pass

    timeline.append({
        "phase": "after_eval",
        "ts": datetime.utcnow().isoformat() + "Z",
        "sys": sample_system(),
        "gpus": sample_nvidia_smi()
    })
    for t in timeline:
        sys_info = t.get("sys")
        if sys_info and sys_info.get("process_rss_bytes") is not None:
            peak_cpu_rss = max(peak_cpu_rss, sys_info["process_rss_bytes"])

        gpus = t.get("gpus")
        if gpus:
            for g in gpus:
                used_mb = g.get("memory_used_mb")
                if used_mb is not None:
                    peak_gpu_mb = max(peak_gpu_mb, float(used_mb))

    end_dt = datetime.utcnow()
    duration_seconds = (end_dt - start_dt).total_seconds()
    duration_ms = int(duration_seconds * 1000000000)
    cpu_duration = duration_ms
    cpu_min_duration = duration_ms
    cpu_max_duration = duration_ms
    cpu_std_dev = 0.0

    if "gpu_duration_ns" in locals() and gpu_duration_ns is not None:
        gpu_duration = gpu_duration_ns
        gpu_min_duration = gpu_duration_ns
        gpu_max_duration = gpu_duration_ns
        gpu_std_dev = 0.0
    else:
        gpu_duration = None
        gpu_min_duration = None
        gpu_max_duration = None
        gpu_std_dev = None
    vm = psutil.virtual_memory() if psutil else None
    total_ram_kb = vm.total // 1024 if vm else None
    free_ram_kb = vm.free // 1024 if (vm and hasattr(vm, "free")) else None
    available_ram_kb = vm.available // 1024 if vm else None
    cached_kb = getattr(vm, 'cached', 0) // 1024 if (vm and hasattr(vm, "cached")) else None


    cpu_usage_kb = None
    if psutil:
        try:
            cpu_time_ns = None
            cpu_vals = [
                t["sys"]["cpu_percent"]
                for t in timeline
                if t.get("phase") == "during_eval"
                   and t.get("sys")
                   and t["sys"].get("cpu_percent") is not None
            ]
            if cpu_vals:
                cpu_time_ns = max(cpu_vals)
        except Exception:
            cpu_usage_kb = None

    gpu_usage_kb = None
    if torch.cuda.is_available() and not force_cpu:
        try:
            torch.cuda.synchronize()
            d = torch.cuda.current_device()
            gpu_usage_kb = f"{int(torch.cuda.memory_allocated(d) / 1024)} kB"
        except Exception:
            gpu_usage_kb = None

    cpu_cores = psutil.cpu_count(logical=True) if psutil else None
    processors = [{"vendor_id": platform.processor() or None, "model": platform.machine() or None}]

    final_report = {
        "model_name": model_name,
        "device_type": get_device_type(),
        "os_version": platform.platform(),
        "valid": len(errors) == 0,
        "emulator": False,
        "iterations": num_batches,
        "error_message": errors[0] if errors else None,
        "unit": torch.cuda.get_device_name(0) if device.type == "cuda" else (platform.processor() or "CPU"),
        "duration": duration_ms,
        "cpu_duration": cpu_duration,
        "cpu_min_duration": cpu_min_duration,
        "cpu_max_duration": cpu_max_duration,
        "cpu_std_dev": cpu_std_dev,

        "gpu_duration": gpu_duration,
        "gpu_min_duration": gpu_min_duration,
        "gpu_max_duration": gpu_max_duration,
        "gpu_std_dev": gpu_std_dev,
        "total_ram_kb": total_ram_kb,
        "free_ram_kb": free_ram_kb,
        "available_ram_kb": available_ram_kb,
        "cached_kb": cached_kb,
        "in_dim_0": in_dim_0,
        "in_dim_1": in_dim_1,
        "in_dim_2": in_dim_2,
        "in_dim_3": in_dim_3,

        "device_analytics": {
            "timestamp": start_epoch,
            "cpu_info": {
                "cpu_cores": cpu_cores,
                "processors": processors,
                "arm_architecture": None
            },
            "timeline": timeline,
            "profile": {
                "top_ops": [],
                "profiler_traces": [],
                "peak_cpu_rss_bytes": peak_cpu_rss,
                "peak_gpu_mb": peak_gpu_mb,
               # "metric_result": metric_result
            }
        }
    }

    ensure_outdir(outpath)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, default=str)

    print(f"[inference_profiler] saved: {outpath}")
    return outpath