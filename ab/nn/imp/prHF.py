#!/usr/bin/env python3

import sys
import os
import json
import re
import argparse
import shutil
import time
import gc
import urllib.request
import tarfile
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.util
import importlib.util
import traceback
import multiprocessing


# --- CONFIGURATION ---
SOURCE_REPO = "NN-Dataset/checkpoints-epoch-50"
TARGET_REPO = "NN-Dataset/pt"  # Base repo, will create subfolder structure
DEFAULT_HF_TOKEN = "" # Hf_token paste here
# --------------------- 

# --- 1. SETUP PATHS ---
script_path = Path(__file__).resolve()
dataset_root = script_path.parents[3]

if str(dataset_root) not in sys.path:
    sys.path.insert(0, str(dataset_root))

# --- WORK DIRS ---
work_dir = dataset_root / "_work"
out_dir = script_path.parent
data_root = work_dir / "prune_data"
temp_dl_dir = work_dir / "temp_prune"
models_dir = dataset_root / "ab" / "nn" / "nn"
transforms_dir = dataset_root / "ab" / "nn" / "transform"

# Target path structure (mirroring quantization)
TARGET_PATH = "structured_l1_layrewise/img-classification_cifar-10_acc"
local_prune_dir = work_dir / "work_prune"
local_prune_dir.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = out_dir / "prune_upload_history.json"
SKIPPED_FILE = out_dir / "prune_skipped_models.json"
ALL_MODELS_JSON = local_prune_dir / "all_models.json"

for p in [data_root, temp_dl_dir]:
    p.mkdir(parents=True, exist_ok=True)

# --- 2. IMPORTS ---
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import torch_pruning as tp
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, upload_file, create_repo

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRUNING_RATIO = 0.30
BATCH_SIZE = 32
MAX_BATCHES = 100
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# ================= DATA HANDLING =================

def download_cifar10():
    """Download CIFAR-10 dataset if not present."""
    os.makedirs(data_root, exist_ok=True)
    tar_path = data_root / "cifar-10-python.tar.gz"
    if not tar_path.exists():
        print(f"   [DATA] Downloading CIFAR-10...")
        urllib.request.urlretrieve(CIFAR_URL, tar_path)

    extract_path = data_root / "cifar-10-batches-py"
    if not extract_path.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_root)
    return extract_path

def get_cifar10_loader():
    """Create a data loader for CIFAR-10 test set."""
    batch_file = data_root / "cifar-10-batches-py" / "test_batch"
    with open(batch_file, "rb") as f:
        data = pickle.load(f, encoding="bytes")

    x = data[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(data[b"labels"])

    for i in range(0, len(x), BATCH_SIZE):
        yield torch.tensor(x[i:i+BATCH_SIZE]), torch.tensor(y[i:i+BATCH_SIZE])

def evaluate_accuracy(model):
    """Evaluate model accuracy on CIFAR-10 test set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(get_cifar10_loader()):
            if i >= MAX_BATCHES:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0

def measure_inference_time(model):
    """Measure average inference time."""
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
    
    # Measurement
    start_time = time.time()
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(get_cifar10_loader()):
            if i >= 10:
                break
            x = x.to(DEVICE)
            model(x)
            count += x.size(0)
    
    return (time.time() - start_time) / count

def count_params(model):
    """Count number of parameters in model."""
    return sum(p.numel() for p in model.parameters())

def model_size_kb(model):
    """Calculate model size in KB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024

# ================= MODEL LOADING =================

def get_resolution_from_transform_file(transform_name: str) -> int:
    """Extract resolution from transform file."""
    if not transform_name:
        return 32
    
    for ext in [".py", ".json"]:
        f = transforms_dir / f"{transform_name}{ext}"
        if f.exists():
            try:
                content = f.read_text()
                match = re.search(r"(?:Resize|size|Crop).*?(\d+)", content, re.IGNORECASE)
                if match:
                    return int(match.group(1))
            except:
                pass
    return 32

def load_model_code(model_name: str):
    """Load model architecture from Python file."""
    # Try direct path
    path = models_dir / f"{model_name}.py"
    if not path.exists():
        # Try without extension
        path = models_dir / f"{model_name}"
        if not path.exists() or path.is_dir():
            return None
    
    spec = importlib.util.spec_from_file_location(model_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def instantiate_model(Net, prm=None):
    """Instantiate model with default parameters."""
    default_prm = {
        'lr': 0.001,
        'momentum': 0.9,
        'dropout': 0.0,
        'dropout_aux': 0.0,
        'batch': 32,
        'epoch': 50,
        'wd': 0.0,
        'norm_eps': 1e-5,
        'norm_momentum': 0.1,
    }
    if prm:
        default_prm.update(prm)
    return Net((1, 3, 32, 32), (10,), default_prm, torch.device(DEVICE))

# ================= PRUNING FUNCTION =================

def apply_pruning(model, example_input, num_classes=10) -> Tuple[Optional[nn.Module], List[str]]:
    """Apply structured L1 layer-wise pruning to the model."""
    try:
        DG = tp.DependencyGraph().build_dependency(model, example_input)
    except Exception as e:
        print(f"   [ERROR] DependencyGraph failed: {e}")
        return None, ["dependency_graph_failed"]

    pruned_methods = []
    conv_count = sum(isinstance(m, nn.Conv2d) for m in model.modules())
    conv_idx = 0
    failed_layers = 0

    all_modules = list(model.modules())

    for m in all_modules:
        # ===== CONVOLUTION PRUNING =====
        if isinstance(m, nn.Conv2d):
            conv_idx += 1

            # Skip grouped & depthwise convolutions
            if m.groups > 1:
                continue

            # Layer-wise pruning ratios
            if conv_idx <= conv_count * 0.3:
                layer_ratio = 0.10  # early layers
            elif conv_idx <= conv_count * 0.7:
                layer_ratio = 0.30  # middle layers
            else:
                layer_ratio = 0.20  # late layers

            w = m.weight.data
            out_channels = w.shape[0]
            num_pruned = int(out_channels * layer_ratio)

            if num_pruned <= 0:
                continue

            # L1 norm based pruning
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
                    pruned_methods.append("conv_l1_structured")
            except Exception:
                failed_layers += 1

        # ===== LINEAR PRUNING =====
        if isinstance(m, nn.Linear):
            # Skip final classifier
            if m.out_features == num_classes:
                continue

            w = m.weight.data
            num_pruned = int(w.shape[0] * 0.30)

            if num_pruned <= 0:
                continue

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
                    pruned_methods.append("linear_l1_structured")
            except Exception:
                failed_layers += 1

    if failed_layers > 5:
        print("   [ERROR] Too many pruning failures")
        return None, ["pruning_unstable_too_many_failures"]

    return model, list(set(pruned_methods))

# ================= UPLOAD HELPERS =================

def upload_with_retry(file_path, repo_path, repo_id, token=None):
    """Upload file with rate limit handling - matches quantization pipeline style."""
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    
    while True:
        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"    Uploaded: {repo_path}")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "too many requests" in error_msg:
                print(f"\n   [WARN] Rate limit hit! Sleeping for 65 minutes...")
                time.sleep(65 * 60)
                continue
            else:
                print(f"    Upload failed: {e}")
                raise e

def slug(s: str) -> str:
    """Create safe filename slug - matches quantization pipeline."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())[:200]

def load_json_safe(path: Path):
    """Safely load JSON file - matches quantization pipeline."""
    if not path.exists() or path.stat().st_size == 0:
        return {} if path.suffix == '.json' else []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {} if path.suffix == '.json' else []

def mark_as_done(name: str):
    """Mark model as processed in history - matches quantization pipeline."""
    current = load_json_safe(HISTORY_FILE)
    if isinstance(current, list):
        if name not in current:
            current.append(name)
    else:
        current = [name]
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(current, f, indent=2)

def log_skip(name: str, reason: str):
    """Log skipped model - matches quantization pipeline."""
    current = load_json_safe(SKIPPED_FILE)
    if not isinstance(current, list):
        current = []
    
    if not any(d.get("model") == name for d in current):
        current.append({"model": name, "reason": reason})
        with open(SKIPPED_FILE, "w") as f:
            json.dump(current, f, indent=2)
    print(f"   [SKIP] {reason}")

# ================= MAIN PROCESSING FUNCTION =================

def process_single_model(model_name: str, args) -> Dict[str, Any]:
    """Process a single model: download, prune, evaluate, save - matches quantization structure."""
    result = {
        "status": "success",
        "accuracy": 0.0,
        "duration": 0,
        "pruning_ratio": PRUNING_RATIO,
        "params_before": 0,
        "params_after": 0,
        "params_removed": 0,
        "model_size_before_kb": 0.0,
        "model_size_after_kb": 0.0
    }
    
    # Create model-specific temp directory (like quantization pipeline)
    m_dir = out_dir / f"tmp_run_{slug(model_name)}"
    m_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"\n� Processing {model_name}...")
        
        # Download checkpoint
        ckpt_path = hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=f"{model_name}.pth",
            cache_dir=str(temp_dl_dir)
        )
        
        # Load model architecture
        module = load_model_code(model_name)
        if not module or not hasattr(module, "Net"):
            raise Exception("Model architecture not found")
        
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        
        # Get transform info from source dataset
        try:
            # Try to get transform from all_models.json in source
            source_json = hf_hub_download(
                repo_id=SOURCE_REPO,
                filename="all_models.json",
                cache_dir=str(temp_dl_dir)
            )
            with open(source_json, 'r') as f:
                source_data = json.load(f)
            transform_name = source_data.get(model_name, {}).get("transform", "default")
        except:
            transform_name = "default"
        
        # Instantiate model
        model = instantiate_model(module.Net)
        model.load_state_dict(sd, strict=False)
        model = model.to(DEVICE)
        
        # Record pre-pruning metrics
        params_before = count_params(model)
        size_before_kb = model_size_kb(model)
        
        # Move model to CPU for safe pruning to prevent CUDA asserts from index out-of-bounds
        model.cpu()
        example_input_cpu = torch.randn(1, 3, 32, 32)
        
        # Apply pruning safely on CPU
        pruned_model_cpu, methods = apply_pruning(model, example_input_cpu, num_classes=10)
        
        if pruned_model_cpu is None:
            raise Exception(f"Pruning failed: {methods[0] if methods else 'unknown'}")
        
        
        # Move back to GPU for evaluation
        pruned_model = pruned_model_cpu.to(DEVICE)
        
        # Post-pruning metrics
        params_after = count_params(pruned_model)
        size_after_kb = model_size_kb(pruned_model)
        accuracy = evaluate_accuracy(pruned_model)
        inf_time = measure_inference_time(pruned_model)
        
        # Prepare result - matches structure in quantization pipeline's JSON
        result.update({
            "status": "success",
            "accuracy": round(accuracy, 4),
            "duration": int(inf_time * 1e9),
            "pruning_ratio": PRUNING_RATIO,
            "params_before": params_before,
            "params_after": params_after,
            "params_removed": params_before - params_after,
            "model_size_before_kb": round(size_before_kb, 2),
            "model_size_after_kb": round(size_after_kb, 2)
        })
        
        # Save pruned model as .pf in temp dir (like quantization saves .tflite)
        pf_path = m_dir / f"{slug(model_name)}.pf"
        
        # Save model state
        torch.save({
            'model_state_dict': pruned_model.state_dict(),
            'pruning_method': 'structured_l1_layerwise',
            'pruning_ratio': PRUNING_RATIO,
            'architecture': model_name,
            'metrics': result
        }, pf_path)
        
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Parameters: {params_before} � {params_after}")
        print(f"    Size: {size_before_kb:.2f}KB � {size_after_kb:.2f}KB")
        
        # --- STRUCTURED UPLOAD - Exactly matching quantization pattern ---
        # 1. Save .pf locally for the user
        final_pf_path = local_prune_dir / f"{slug(model_name)}.pf"
        shutil.copy2(pf_path, final_pf_path)
        
        if args.push_hf:
            print(f"   [UPLOAD] Syncing to Hugging Face...")
            
            # 2. Upload .pf file to the structured path
            # Path: structured_l1_layerwise/img-classification_cifar-10_acc/ModelName.pf
            upload_with_retry(
                pf_path,
                f"{TARGET_PATH}/{slug(model_name)}.pf",
                TARGET_REPO,
                token=args.hf_token
            )
        
        # Mark as done
        mark_as_done(model_name)
        
    except Exception as e:
        result["status"] = "failed"
        log_skip(model_name, str(e))
        traceback.print_exc()
        if "device-side assert" in str(e):
            import sys; sys.exit(1)
        print(f"    Failed: {e}")
    
    finally:
        # Cleanup temp directory (like quantization pipeline)
        shutil.rmtree(m_dir, ignore_errors=True)
        
        # Cleanup memory
        if 'model' in locals():
            del model
        if 'pruned_model' in locals():
            del pruned_model
        gc.collect()
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except:
                pass
    
    return result

# ================= MAIN =================

def _process_worker(model_name, args, q):
    try:
        result = process_single_model(model_name, args)
        q.put(("SUCCESS", result))
    except Exception as e:
        q.put(("ERROR", {
            "status": "failed",
            "accuracy": 0.0,
            "duration": 0,
            "pruning_ratio": PRUNING_RATIO,
            "params_before": 0,
            "params_after": 0,
            "params_removed": 0,
            "model_size_before_kb": 0.0,
            "model_size_after_kb": 0.0
        }))

def main():
    parser = argparse.ArgumentParser(description="Prune and upload models - matches quantization structure")
    parser.add_argument("--push-hf", action="store_true", help="Upload to Hugging Face")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from last processed")
    parser.add_argument("--hf-token", default=DEFAULT_HF_TOKEN, help="Hugging Face token")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of models to process")
    parser.add_argument("--model", type=str, default=None, help="Process specific model only")
    args = parser.parse_args()
    
    # Set token
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    print("=" * 60)
    print("PRUNING PIPELINE - Structured L1 Layer-wise")
    print("=" * 60)
    print(f"Source: {SOURCE_REPO}")
    print(f"Target: {TARGET_REPO}/{TARGET_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Pruning Ratio: {PRUNING_RATIO}")
    print("=" * 60)
    
    # Download CIFAR-10
    print("\n= Preparing CIFAR-10 dataset...")
    download_cifar10()
    # Get list of models
    print(f"\nScanning source repository...")
    all_files = list_repo_files(SOURCE_REPO)
    pth_files = [f.replace(".pth", "") for f in all_files if f.endswith(".pth")]

    # Filter mathematically exactly the image classification models directly via HF all_models.json
    try:
        source_json = hf_hub_download(
            repo_id=SOURCE_REPO,
            filename="all_models.json",
            cache_dir=str(temp_dl_dir)
        )
        with open(source_json, 'r') as f:
            source_data = json.load(f)
        
        pth_files = [
            f for f in pth_files 
            if getattr(source_data.get(f, {}), "get", lambda x: None)("task") == "img-classification"
        ]
    except Exception as e:
        print(f"  Warning: Could not fetch HF all_models.json for accurate filtering: {e}")
        # fallback to basic generic filter if download fails
        excluded_prefixes = ("rag-", "txt-", "obj-", "seg-", "lstm", "rnn")
        pth_files = [f for f in pth_files if not f.lower().startswith(excluded_prefixes)]
        
    print(f"  Filtered down to {len(pth_files)} image classification models.")    
    pth_files.sort()  # Sort for consistent processing
    
    if args.model:
        pth_files = [args.model] if args.model in pth_files else []
        if not pth_files:
            print(f" Model '{args.model}' not found in source")
            return
    
    if args.limit:
        pth_files = pth_files[:args.limit]
    
    print(f" Found {len(pth_files)} models to process")
    
    # Load history
    processed = set()
    if args.resume and HISTORY_FILE.exists():
        history = load_json_safe(HISTORY_FILE)
        if isinstance(history, list):
            processed = set(history)
            print(f" Resume mode: {len(processed)} models already processed")
    # Process models
    results = load_json_safe(ALL_MODELS_JSON) if ALL_MODELS_JSON.exists() else {}
    successful = 0
    failed = 0
    
    for i, model_name in enumerate(pth_files, 1):
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(pth_files)}] Processing: {model_name}")
        print(f"{'='*60}")
        
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_process_worker, args=(model_name, args, q))
        p.start()
        
        # Wait up to 3 minutes (0 seconds)
        p.join(180)
        
        if p.is_alive():
            print(f"   [TIMEOUT] Model {model_name} processing exceeded 5 minutes. Killing process...")
            p.terminate()
            p.join()
        else:
            if not q.empty():
                status, res = q.get()
                result = res # could be SUCCESS or ERROR dict
            else:
                result = {
                    "status": "failed",
                    "accuracy": 0.0,
                    "duration": 0,
                    "pruning_ratio": PRUNING_RATIO,
                    "params_before": 0,
                    "params_after": 0,
                    "params_removed": 0,
                    "model_size_before_kb": 0.0,
                    "model_size_after_kb": 0.0
                }
        
        if result.get("status") == "success":
            results[model_name] = result
            successful += 1
        else:
            # Only store minimal failure info
            results[model_name] = {
                "status": "failed"
            }
            failed += 1
            
        # Update comprehensive all_models.json incrementally ensuring all metadata is captured
        with open(ALL_MODELS_JSON, 'w') as f:
            json.dump(results, f, indent=2)
            
        if args.push_hf:
            upload_with_retry(
                ALL_MODELS_JSON,
                f"{TARGET_PATH}/all_models.json",
                TARGET_REPO,
                token=args.hf_token
            )
        
        # Small delay between models (like quantization pipeline)
        if i < len(pth_files):
            print("\n Cooling down before next model...")
            time.sleep(3)
    
    # Generate summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    with open(ALL_MODELS_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n= All metrics saved correctly to: {ALL_MODELS_JSON}")
    
    if args.push_hf:
        print(f"\n= All files uploaded to: https://huggingface.co/{TARGET_REPO}/tree/main/{TARGET_PATH}")
        print(f"   - Individual models: {TARGET_PATH}/${'model_name'}.pf")
        print(f"   - Metrics: {TARGET_PATH}/all_models.json")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
