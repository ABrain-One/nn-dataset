import os
import json
import tarfile
import pickle
import urllib.request
from pathlib import Path
import importlib.util

import torch
import torch.nn as nn
import numpy as np
import torch_pruning as tp
from huggingface_hub import hf_hub_download, list_repo_files

# ================= CONFIG =================

HF_REPO_ID = "NN-Dataset/checkpoints-epoch-50"
HF_SUMMARY_FILE = "all_models_summary.json"

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = SCRIPT_DIR.parents[2] / "nn"
OUTPUT_DIR = SCRIPT_DIR.parents[1] / "stat" / "imp"
OUTPUT_JSON = OUTPUT_DIR / "prune.json"

DEVICE = "cuda"
PRUNING_RATIO = 0.30

BATCH_SIZE = 32
MAX_BATCHES = 20

DATA_DIR = SCRIPT_DIR / "data"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# ================= DATA =================

def download_cifar10():
    os.makedirs(DATA_DIR, exist_ok=True)
    tar_path = DATA_DIR / "cifar-10-python.tar.gz"
    if not tar_path.exists():
        urllib.request.urlretrieve(CIFAR_URL, tar_path)

    extract_path = DATA_DIR / "cifar-10-batches-py"
    if not extract_path.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(DATA_DIR)

def cifar10_loader():
    batch_file = DATA_DIR / "cifar-10-batches-py" / "test_batch"
    with open(batch_file, "rb") as f:
        data = pickle.load(f, encoding="bytes")

    x = data[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(data[b"labels"])

    for i in range(0, len(x), BATCH_SIZE):
        yield torch.tensor(x[i:i+BATCH_SIZE]), torch.tensor(y[i:i+BATCH_SIZE])

def evaluate_accuracy(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(cifar10_loader()):
            if i >= MAX_BATCHES:
                break
            out = model(x.to(DEVICE))
            if isinstance(out, tuple):
                out = out[0]
            pred = out.argmax(dim=1)
            correct += (pred.cpu() == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0

# ================= MODEL =================

def load_model_code(model_name):
    # Try direct path
    path = LOCAL_MODEL_DIR / f"{model_name}.py"
    if not path.exists():
        # Try subdirectory 'nn' where some RAG models might be
        path = LOCAL_MODEL_DIR / "nn" / f"{model_name}.py"
        if not path.exists():
            return None
            
    spec = importlib.util.spec_from_file_location(model_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def instantiate_model(Net):
    prm = {
        'lr': 0.001, 
        'momentum': 0.9, 
        'dropout': 0.0,
        'dropout_aux': 0.0,  # Required by GoogLeNet
        'batch': 32,
        'epoch': 50,
        'wd': 0.0,
        'norm_eps': 1e-5,       # Required by MobileNetV3
        'norm_momentum': 0.1,   # Required by MobileNetV3
    }
    return Net((1,3,32,32), (10,), prm, torch.device(DEVICE))

import time
import gc

# ... (existing imports)

# ... (existing code)

# ================= REAL PRUNING =================

def physical_prune_model(model, example_input, num_classes=10):
    # Robustness Fix 2: Safe DependencyGraph
    try:
        DG = tp.DependencyGraph().build_dependency(model, example_input)
    except Exception as e:
        print(f"  ❌ DependencyGraph failed: {e}")
        return None, ["dependency_graph_failed"]

    pruned_layers = []
    conv_count = sum(isinstance(m, nn.Conv2d) for m in model.modules())
    conv_idx = 0
    
    # Robustness Fix 3: Failure counter
    failed_layers = 0

    print("\n  [Layer-wise Pruning Schedule Applied]")
    
    # Get standard modules list for indexing safe checks
    all_modules = list(model.modules())

    for m in all_modules:

        # ===================== CONVOLUTION =====================
        if isinstance(m, nn.Conv2d):
            conv_idx += 1

            # ❌ Skip grouped & depthwise convolutions (RegNet, MobileNet safe)
            if m.groups > 1:
                print("  - Skipping grouped Conv2d")
                continue

            # ---- Layer-wise pruning ratio ----
            if conv_idx <= conv_count * 0.3:
                layer_ratio = 0.10   # early layers
            elif conv_idx <= conv_count * 0.7:
                layer_ratio = 0.30   # middle layers
            else:
                layer_ratio = 0.20   # late layers

            w = m.weight.data
            out_channels = w.shape[0]
            num_pruned = int(out_channels * layer_ratio)

            if num_pruned <= 0:
                continue

            print(f"  - Conv2d [{conv_idx}/{conv_count}] prune {num_pruned}/{out_channels} ({layer_ratio:.0%})")

            l1_norm = w.abs().sum(dim=(1, 2, 3))
            _, pruning_idxs = torch.topk(l1_norm, k=num_pruned, largest=False)
            pruning_idxs = pruning_idxs.tolist()

            try:
                if hasattr(DG, "get_pruning_group"):
                    plan = DG.get_pruning_group(m, tp.prune_conv_out_channels, idxs=pruning_idxs)
                else:
                    plan = DG.get_pruning_plan(m, tp.prune_conv_out_channels, idxs=pruning_idxs)
                
                if plan:
                    plan.exec()
                    pruned_layers.append("conv_l1_structured")
            except Exception:
                failed_layers += 1
                pass

        # ===================== LINEAR =====================
        if isinstance(m, nn.Linear):

            # Robustness Fix 4: Classifier-safe pruning (robust)
            if m.out_features == num_classes:
                print(f"  - Skipping final classifier (out={m.out_features})")
                continue

            w = m.weight.data
            num_pruned = int(w.shape[0] * 0.30)

            if num_pruned <= 0:
                continue

            print(f"  - Linear prune {num_pruned}/{w.shape[0]} (30%)")

            l1_norm = w.abs().sum(dim=1)
            _, pruning_idxs = torch.topk(l1_norm, k=num_pruned, largest=False)
            pruning_idxs = pruning_idxs.tolist()

            try:
                if hasattr(DG, "get_pruning_group"):
                    plan = DG.get_pruning_group(m, tp.prune_linear_out_channels, idxs=pruning_idxs)
                else:
                    plan = DG.get_pruning_plan(m, tp.prune_linear_out_channels, idxs=pruning_idxs)

                if plan:
                    plan.exec()
                    pruned_layers.append("linear_l1_structured")
            except Exception:
                failed_layers += 1
                pass

    # Robustness Fix 3: Abort if too many failures
    if failed_layers > 5:
        print("  ❌ Too many pruning failures, aborting model")
        return None, ["pruning_unstable_too_many_failures"]

    return model, list(set(pruned_layers))

# ================= METRICS =================

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def model_size_kb(model):
    # Convert bytes to Kilobytes
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024

def measure_inference_time(model):
    model.eval()
    # Warmup
    dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE)
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
            
    # Measurement
    start_time = time.time()
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(cifar10_loader()):
            if i >= 10: # Measure over 10 batches for speed estimation
                break
            x = x.to(DEVICE)
            model(x)
            count += x.size(0)
            
    end_time = time.time()
    return end_time - start_time

DEVICE = "cuda"

# ================= HELPER =================

def get_model_paths(model_name):
    # Determine base directory based on model type
    if model_name.lower().startswith("rag"):
        # RAG models go to .../stat/imp (User requested same folder)
        start_dir = OUTPUT_DIR
        # RAG Naming Convention: sl1c_rag_cifar-10_acc_{model}.json (No 'classification')
        json_filename = f"sl1c_rag_cifar-10_acc_{model_name}.json"
    else:
        # Standard models go to .../stat/imp (no subfolders)
        start_dir = OUTPUT_DIR
        # Naming convention encoded in filename to avoid subfolders
        json_filename = f"sl1c_img-classification_cifar-10_acc_{model_name}.json"
        
    os.makedirs(start_dir, exist_ok=True)
    
    json_path = start_dir / json_filename
    return start_dir, json_path

# ================= MAIN =================

import multiprocessing

# ================= WORKER =================

def process_single_model(pth, file_index, total_files):
    try:
        model_name = pth.replace(".pth", "")
        
        # Use new dynamic path helper
        output_dir, model_output_json = get_model_paths(model_name)
        
        print(f"\n[{file_index}/{total_files}] REAL PRUNING → {model_name}")
        
        # Initial metadata
        meta = {"model": model_name, "dataset": "cifar-10", "task": "img-classification"}
        
        # Check if this is a fresh start (not resuming) and create initial entry
        if not model_output_json.exists():
            current_result = {
                **meta,
                "status": "processing",
                "timestamp_start": time.time()
            }
            with open(model_output_json, "w") as f:
                json.dump(current_result, f, indent=4)

        module = load_model_code(model_name)
        if not module or not hasattr(module, "Net"):
            print("❌ model code missing")
            current_result = {
                **meta,
                "status": "skipped",
                "error": "model code not found locally"
            }
            with open(model_output_json, "w") as f:
                json.dump(current_result, f, indent=4)
            return "code_missing"

        try:
            model = instantiate_model(module.Net)
            
            ckpt = hf_hub_download(HF_REPO_ID, f"{model_name}.pth")
            sd = torch.load(ckpt, map_location=DEVICE)
            sd = sd.get("state_dict", sd)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            model = model.to(DEVICE) # Ensure model is on correct device
        except Exception as e:
            print(f"❌ weight load failed: {e}")
            current_result = {
                **meta,
                "status": "failed",
                "error": f"checkpoint fetch/load failed: {str(e)}"
            }
            with open(model_output_json, "w") as f:
                json.dump(current_result, f, indent=4)
            return "failed"

        try:
            params_before = count_params(model)
            size_before_kb = model_size_kb(model)
            
            example_input = torch.randn(1,3,32,32).to(DEVICE)

            # Pruning
            model, methods = physical_prune_model(model, example_input, num_classes=10)
            
            if model is None:
                error_msg = methods[0] if methods else "unknown pruning failure"
                print(f"❌ Pruning Aborted: {error_msg}")
                current_result = {
                    **meta,
                    "status": "skipped",
                    "error": error_msg
                }
                with open(model_output_json, "w") as f:
                    json.dump(current_result, f, indent=4)
                return "failed"

            else:
                params_after = count_params(model)
                size_after_kb = model_size_kb(model)
                
                # Inference Time & Accuracy
                try:
                    acc = evaluate_accuracy(model)
                    inf_time = measure_inference_time(model)
                    
                    # Update success results
                    current_result = {
                        "model": meta["model"],
                        "dataset": meta["dataset"],
                        "task": meta["task"],
                        "status": "success",
                        "accuracy_after_pruning": round(acc,4),
                        "inference_time_sec": round(inf_time, 6),
                        "pruning_method": "structured_l1_layerwise",
                        "pruning_ratio": PRUNING_RATIO,
                        "params_before": params_before,
                        "params_after": params_after,
                        "params_removed": params_before - params_after,
                        "model_size_before_kb": round(size_before_kb, 2),
                        "model_size_after_kb": round(size_after_kb, 2)
                    }
                    print(f"✔ Params: {params_before} → {params_after} | Size: {size_after_kb:.2f} KB | Inf: {inf_time:.4f}s")
                    with open(model_output_json, "w") as f:
                        json.dump(current_result, f, indent=4)
                    return "processed"
                
                except Exception as e:
                    print(f"❌ Eval failed: {e}")
                    current_result = {
                        **meta,
                        "status": "failed",
                        "error": "eval failed"
                    }
                    with open(model_output_json, "w") as f:
                        json.dump(current_result, f, indent=4)
                    return "failed"
        
        except RuntimeError as e:
            print(f"❌ Pruning/Eval failed (RuntimeError): {e}")
            current_result = {
                **meta,
                "status": "failed",
                "error": str(e)
            }
            with open(model_output_json, "w") as f:
                json.dump(current_result, f, indent=4)
            return "failed"

        except Exception as e:
            print(f"❌ Pruning/Eval failed (Exception): {e}")
            current_result = {
                **meta,
                "status": "failed",
                "error": str(e)
            }
            with open(model_output_json, "w") as f:
                json.dump(current_result, f, indent=4)
            return "failed"
            
    except Exception as e:
         print(f"🚨 CRITICAL ERROR processing {model_name}: {e}")
         return "failed"
    finally:
        if 'model' in locals():
            del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# ================= MAIN =================

def main():
    # Remove static makedirs, handled dynamically
    download_cifar10()

    print(f"🔍 Scanning HF Repo {HF_REPO_ID} for all .pth files...")
    
    # Strictly list all files in repo
    all_files = list_repo_files(HF_REPO_ID)
    pth_files = [f for f in all_files if f.endswith(".pth")]
    
    print(f"✔ Found {len(pth_files)} models to prune.")
    
    stats = {"rag": 0, "resume": 0, "processed": 0, "failed": 0, "code_missing": 0}

    for i, pth in enumerate(pth_files, 1):
        model_name = pth.replace(".pth", "")
        
        # Check if already processed
        _, model_output_json = get_model_paths(model_name)
        if model_output_json.exists():
            try:
                with open(model_output_json, "r") as f:
                    data = json.load(f)
                
                # Check for stuck 'processing' files
                if data.get("status") == "processing":
                    # Check age
                    start_time = data.get("timestamp_start", 0)
                    if time.time() - start_time > 600: # 10 mins old = stuck
                        print(f"⚠️  Found STUCK 'processing' file for {model_name} (>10m old). Marking as SKIPPED.")
                        
                        # Ensure metadata exists
                        data.update({
                            "model": model_name,
                            "dataset": "cifar-10",
                            "task": "img-classification"
                        })

                        data["status"] = "skipped"
                        data["error"] = "Previous run stuck/frozen (timeout)"
                        with open(model_output_json, "w") as f:
                            json.dump(data, f, indent=4)
                        continue
                    else:
                        print(f"⏳ Skipping {model_name} (currently processing elsewhere)")
                        continue
                else:
                    print(f"✔ Skipping {model_name} (already pruned)")
                    continue
            except json.JSONDecodeError:
                print(f"⚠️  Corrupt JSON for {model_name}. Re-running...")
                model_output_json.unlink()

        # Subprocess Isolation with Timeout
        p = multiprocessing.Process(target=process_single_model, args=(pth, i, len(pth_files)))
        p.start()
        p.join(timeout=180) # Wait for completion with shorter timeout (3 mins)
        
        if p.is_alive():
            print(f"⏳ Model {model_name} timed out (>10m). Terminating...")
            p.terminate()
            p.join()
            stats["failed"] += 1
            # Log timeout
            output_dir, model_output_json = get_model_paths(model_name)
            if model_output_json.exists():
                try:
                    with open(model_output_json, "r") as f:
                        data = json.load(f)
                    
                    # Ensure metadata exists
                    data.update({
                        "model": model_name,
                        "dataset": "cifar-10",
                        "task": "img-classification"
                    })
                    
                    data["status"] = "timed_out"
                    data["error"] = "Process timed out (>180s)"
                    with open(model_output_json, "w") as f:
                        json.dump(data, f, indent=4)
                except:
                    pass
        elif p.exitcode != 0:
            print(f"❌ Model {model_name} crashed (Exit code: {p.exitcode}). Skipping...")
            stats["failed"] += 1
            # Try to log crash if file exists
            output_dir, model_output_json = get_model_paths(model_name)
            if model_output_json.exists():
                try:
                    with open(model_output_json, "r") as f:
                        data = json.load(f)
                    
                    # Ensure metadata exists
                    data.update({
                        "model": model_name,
                        "dataset": "cifar-10",
                        "task": "img-classification"
                    })

                    data["status"] = "crashed"
                    data["error"] = "Process killed (OOM or Segfault)"
                    with open(model_output_json, "w") as f:
                        json.dump(data, f, indent=4)
                except:
                    pass

    print(f"\n✅ BATCH PROCESSING COMPLETE")
    # Note: Stats are harder to track perfectly with processes without a queue, 
    # but the logs will show it.
    # === Consolidate Results ===
    summary_list = []
    print(f"\n📊 Generating summary file: prune_all_models_summary.json ...")
    
    for json_file in OUTPUT_DIR.glob("*.json"):
        if json_file.name == "prune.json": continue # Skip config output
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                if "baseline_accuracy" in data:
                    del data["baseline_accuracy"]
                
                # Check for missing metadata (fallback for legacy failures)
                if "model" not in data:
                    fname = json_file.name
                    if fname.startswith("sl1c_img-classification_cifar-10_acc_"):
                        data["model"] = fname.replace("sl1c_img-classification_cifar-10_acc_", "").replace(".json", "")
                    elif fname.startswith("sl1c_rag_cifar-10_acc_"):
                        data["model"] = fname.replace("sl1c_rag_cifar-10_acc_", "").replace(".json", "")
                
                if "dataset" not in data: 
                    data["dataset"] = "cifar-10"
                if "task" not in data: 
                    data["task"] = "img-classification"
                    
                summary_list.append(data)
        except Exception as e:
            print(f"⚠️  Error reading {json_file.name}: {e}")
            
    summary_path = OUTPUT_DIR / "prune_all_models_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_list, f, indent=4)
        
    print(f"✅ Saved summary for {len(summary_list)} models to {summary_path}")

    print(f"SUMMARY: Check individual .json files in stat/imp/")

if __name__ == "__main__":
    # Fix for spawn method if needed
    multiprocessing.set_start_method('spawn', force=True)
    main()
