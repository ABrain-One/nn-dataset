#!/usr/bin/env python3
"""
run.py — inference profiler that uses existing Train.Train.eval(test_loader) when available.
- CLI: python -m ab.nn.run --model-class ab.nn.nn.ComplexNet.Net --dataset cifar-10 --config img-classification --no-profiler
- Output folder: ab/nn/stat/run/<config>_<architecture>-<timestamp> (or <architecture>-<timestamp>)
- Output filename: windows_devicetype.json
"""

import argparse
from ab.nn.util.Util import (
    import_by_path,
    get_device_type,
    sanitize_filename,
    ensure_outdir,
    sanitize_name,
    extract_arch_name,
    default_outpath,
    sample_nvidia_smi,
    sample_system
)
from ab.nn.util.Inference import run_inference
try:
    import psutil
except Exception:
    psutil = None

def main():
    p = argparse.ArgumentParser(prog="ab.nn.run", description="Profile inference and save JSON report.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model-class", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--num-batches", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", dest="outpath", type=str, default=None)
    p.add_argument("--no-profiler", dest="no_profiler", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--force-cpu", action="store_true")
    args = p.parse_args()

    if args.outpath is None:
        model_name = extract_arch_name(args.model_class)
        outpath = default_outpath(model_name, config=args.config)
    else:
        outpath = args.outpath

    run_inference(
        config=args.config,
        model_class=args.model_class,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        outpath=outpath,
        use_profiler=(not args.no_profiler),
        debug=args.debug,
        force_cpu=args.force_cpu
    )


if __name__ == "__main__":
    main()
