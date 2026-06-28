import os
import json
import time
import platform
import argparse
import statistics
import psutil
import torch
import subprocess
from datetime import datetime

from ab.nn.util.Const import stat_run_dir
from ab.nn.util.Util import sample_system, sample_nvidia_smi, get_in_shape, first_tensor
from ab.nn import api
from ab.nn.util.db.Util import get_attr
from ab.nn.util.Loader import load_dataset

OUTPUT_BASE_PATH = stat_run_dir / 'pt' / 'fp32'

def get_device_info():
    os_name = platform.system()
    try:
        if os_name == "Windows":
            result = subprocess.run(
                ["wmic", "computersystem", "get", "Model"],
                capture_output=True, text=True
            )
            product = result.stdout.split("\n")[1].strip()
            vendor = ""
        elif os_name == "Darwin":
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True, text=True
            )
            product = ""
            vendor = ""
            for line in result.stdout.splitlines():
                if "Model Name" in line:
                    product = line.split(":")[1].strip()
        else:  # Linux
            with open("/sys/class/dmi/id/sys_vendor") as f:
                vendor = f.read().strip()
            with open("/sys/class/dmi/id/product_name") as f:
                product = f.read().strip()
        return vendor, product
    except Exception:
        return "", platform.node()

def get_model_params_from_db(model_name, dataset=None):
    try:
        if dataset:
            df_data = api.data(nn=model_name, dataset=dataset, only_best_accuracy=True)
        else:
            df_data = api.data(nn=model_name, only_best_accuracy=True)
        row = df_data.sort_values('accuracy', ascending=False).iloc[0]
        prm = row['prm']
        print(f"  ✓ Loaded from DB: task={row['task']}, dataset={row['dataset']}, "
              f"transform={prm['transform']}, accuracy={row['accuracy']:.4f}")
        return {
            "prm": prm,
            "task": row['task'],
            "dataset": row['dataset'],
            "metric": row['metric']
        }
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        raise

def run_test_drive(model_name, dataset=None):
    print(f"\n{'='*80}")
    print(f"Starting Inference: {model_name}")
    print(f"{'='*80}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("Step 1: Fetching model from database...")
    model_data = get_model_params_from_db(model_name, dataset=dataset)
    exact_prm = model_data["prm"]
    task = model_data["task"]
    dataset = model_data["dataset"]
    metric = model_data["metric"]
    num_workers = exact_prm.get('num_workers', 1)

    print("Step 2: Loading dataset...")
    try:
        out_shape, _, train_set, _ = load_dataset(task, dataset, exact_prm['transform'])
    except Exception as e:
        print(f"✗ ERROR loading dataset: {e}")
        raise

    in_shape = get_in_shape(train_set, num_workers)
    sample_tensor = first_tensor(train_set, num_workers)

    print("Step 3: Loading model...")
    try:
        nn_module = f"ab.nn.nn.{model_name}"
        model_net = get_attr(nn_module, 'Net')
        print(f"  ✓ Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"  ✗ FAILED to load model {model_name}. Error: {e}")
        raise

    print("Step 4: Taking system snapshot (before)...")
    timeline = []
    timeline.append({
        "phase": "before_eval",
        "ts": datetime.utcnow().isoformat() + "Z",
        "sys": sample_system(),
        "gpus": sample_nvidia_smi()
    })

    cpu_model = model_net(
        in_shape=in_shape,
        out_shape=out_shape,
        prm=exact_prm,
        device=torch.device("cpu")
    ).cpu()
    cpu_model.eval()
    cpu_inputs = sample_tensor.cpu()

    print("Step 5: CPU warmup (10 iterations)...")
    with torch.no_grad():
        for _ in range(10):
            _ = cpu_model(cpu_inputs)

    gpu_model = model_net(
        in_shape=in_shape,
        out_shape=out_shape,
        prm=exact_prm,
        device=torch.device("cuda")
    ).cuda()
    gpu_model.eval()
    gpu_inputs = sample_tensor.cuda()

    print("Step 6: GPU warmup (10 iterations)...")
    with torch.no_grad():
        for _ in range(10):
            _ = gpu_model(gpu_inputs)
    torch.cuda.synchronize()

    iterations = 20

    cpu_runs = []
    for _ in range(iterations):
        start_cpu = time.perf_counter()
        with torch.no_grad():
            _ = cpu_model(cpu_inputs)
        end_cpu = time.perf_counter()
        cpu_runs.append(((end_cpu - start_cpu) * 1_000_000_000))

    cpu_duration_ns = int(sum(cpu_runs) / len(cpu_runs))
    cpu_min_duration = int(min(cpu_runs))
    cpu_max_duration = int(max(cpu_runs))
    cpu_std_dev = float(statistics.stdev(cpu_runs)) if len(cpu_runs) > 1 else 0.0

    gpu_runs = []
    for _ in range(iterations):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        with torch.no_grad():
            _ = gpu_model(gpu_inputs)
        ender.record()
        torch.cuda.synchronize()
        gpu_runs.append((starter.elapsed_time(ender) * 1_000_000))

    gpu_duration_ns = int(sum(gpu_runs) / len(gpu_runs))
    gpu_min_duration = int(min(gpu_runs))
    gpu_max_duration = int(max(gpu_runs))
    gpu_std_dev = float(statistics.stdev(gpu_runs)) if len(gpu_runs) > 1 else 0.0

    print("Step 7: Taking system snapshot (after)...")
    timeline.append({
        "phase": "after_eval",
        "ts": datetime.utcnow().isoformat() + "Z",
        "sys": sample_system(),
        "gpus": sample_nvidia_smi()
    })

    peak_cpu_rss = 0
    peak_gpu_mb = 0.0
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

    vendor, product = get_device_info()
    device_type = f"{vendor} {product}".strip() or platform.node()
    vm = psutil.virtual_memory()

    print("Step 8: Building report...")
    final_report = {
        "model_name": model_name,
        "device_type": device_type,
        "os_version": platform.platform(),
        "valid": True,
        "emulator": False,
        "iterations": iterations,
        "error_message": None,
        "unit": gpu_name,
        "duration": cpu_duration_ns,
        "cpu_duration": cpu_duration_ns,
        "cpu_min_duration": cpu_min_duration,
        "cpu_max_duration": cpu_max_duration,
        "cpu_std_dev": cpu_std_dev,
        "gpu_duration": gpu_duration_ns,
        "gpu_min_duration": gpu_min_duration,
        "gpu_max_duration": gpu_max_duration,
        "gpu_std_dev": gpu_std_dev,
        "total_ram_kb": vm.total // 1024,
        "free_ram_kb": vm.free // 1024,
        "available_ram_kb": vm.available // 1024,
        "cached_kb": getattr(vm, 'cached', 0) // 1024,
        "in_dim_0": in_shape[0],
        "in_dim_1": in_shape[2],
        "in_dim_2": in_shape[3],
        "in_dim_3": in_shape[1],
        "device_analytics": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "cpu_info": {
                "cpu_cores": psutil.cpu_count(logical=True),
                "processors": [{"vendor_id": platform.processor(), "model": platform.machine()}],
                "arm_architecture": None
            },
            "timeline": timeline,
            "profile": {
                "top_ops": [],
                "profiler_traces": [],
                "peak_cpu_rss_bytes": peak_cpu_rss,
                "peak_gpu_mb": peak_gpu_mb
            }
        }
    }

    print("Step 9: Saving JSON output...")
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    dataset_folder = f"{task}_{dataset}_{metric}_{model_name}"
    save_dir = os.path.join(OUTPUT_BASE_PATH, dataset_folder)
    os.makedirs(save_dir, exist_ok=True)
    _, product = get_device_info()
    device_file = f"{platform.system()}_{product}".replace(" ", "")
    save_path = os.path.join(save_dir, f"{device_file}.json")

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)
        print(f"  ✓ Saved to: {save_path}")
    except Exception as e:
        print(f"  ✗ ERROR saving: {e}")
        return

    print(f"\n✓ SUCCESS!")
    print(f"  Dataset:        {dataset} | Transform: {exact_prm['transform']}")
    print(f"  Input shape:    {in_shape[2]}x{in_shape[3]} px, {in_shape[1]} ch")
    print(f"  CPU time:       {cpu_duration_ns / 1_000_000:.2f} ms per iteration")
    print(f"  GPU time:       {gpu_duration_ns / 1_000_000:.2f} ms per iteration")
    print(f"  Peak GPU Mem:   {peak_gpu_mb:.2f} MB")
    print(f"  Peak CPU Mem:   {peak_cpu_rss / (1024*1024):.2f} MB")
    print(f"  Output: {save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference testing for 15K+ neural networks")
    parser.add_argument("--model-name", type=str, default=None, help="Single model name")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset (default: best-accuracy row)")
    parser.add_argument("--batch", action="store_true", help="Run batch inference")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of models")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N models")
    args = parser.parse_args()

    if args.batch:
        print("=" * 80)
        print("BATCH MODE: Processing all models from database")
        print(f"Output path: {OUTPUT_BASE_PATH}")
        print("=" * 80)

        try:
            print("\nFetching model list from database...")
            df_all = api.data(max_rows=None)
            unique_models = sorted(df_all['nn'].unique().tolist())

            total = len(unique_models)
            print(f"Found {total} unique models in database\n")

            if args.skip > 0:
                unique_models = unique_models[args.skip:]
                print(f"Skipped first {args.skip} models\n")

            if args.limit:
                unique_models = unique_models[:args.limit]

            print(f"Processing {len(unique_models)} models...\n")

            success_count = 0
            fail_count = 0

            for idx, model_name in enumerate(unique_models, 1):
                print(f"[{idx}/{len(unique_models)}] Processing: {model_name}")
                try:
                    run_test_drive(model_name=model_name, dataset=args.dataset)
                    success_count += 1
                except Exception as e:
                    print(f"  ✗ Error: {e}\n")
                    fail_count += 1
                    continue

            print("\n" + "=" * 80)
            print(f"BATCH COMPLETE: {success_count} successful, {fail_count} failed")
            print(f"Output directory: {OUTPUT_BASE_PATH}")
            print("=" * 80 + "\n")

        except Exception as e:
            print(f"✗ ERROR: {e}")

    elif args.model_name:
        print(f"Output path: {OUTPUT_BASE_PATH}\n")
        run_test_drive(model_name=args.model_name, dataset=args.dataset)

    else:
        print("\nUSAGE:")
        print("  python inference.py --model-name AlexNet")
        print("  python inference.py --model-name AlexNet --dataset cifar-10")
        print("  python inference.py --batch --limit 100")