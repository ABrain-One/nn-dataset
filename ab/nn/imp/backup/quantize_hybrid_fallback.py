import os
import sys
import torch
import torch.nn as nn
import json
import time
import importlib.util
import torchvision
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
import copy

# --- PATH SETUP ---
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
REPO_ID = "NN-Dataset/checkpoints-epoch-5"       
LOCAL_MODEL_DIR = os.path.join("ab", "nn", "nn") 
OUTPUT_FOLDER = "comparison_results"             
FINAL_REPORT_FILE = "QUANTIZATION_COMPARISON_REPORT.json"

# Settings
TEST_BATCH_LIMIT = 50   # for speed efficency
ACCURACY_DROP_LIMIT = 0.10 # ⚠️ more than 10% drop will not be bearable

# ==========================================
# 🍬 STATIC WRAPPER
# ==========================================
class QuantWrapper(nn.Module):
    def __init__(self, model):
        super(QuantWrapper, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_model_class(file_path):
    try:
        spec = importlib.util.spec_from_file_location("dynamic_model", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Net
    except Exception:
        return None

def get_dataloaders():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    calib_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    return calib_loader, test_loader

def evaluate_model(model, data_loader, device="cpu"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            if TEST_BATCH_LIMIT and i >= TEST_BATCH_LIMIT: break
            inputs, labels = inputs.to(device), labels.to(device)
            try:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except Exception:
                return -1.0, 0.0 # Crash indicator
            
    end_time = time.time()
    inference_time = end_time - start_time
    accuracy = correct / total if total > 0 else 0
    return accuracy, inference_time

def get_file_size_mb(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    print(f"\n🚀 STARTING HYBRID REPAIR PIPELINE")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    calib_loader, test_loader = get_dataloaders()
    
    try:
        summary_path = hf_hub_download(repo_id=REPO_ID, filename="all_models_summary.json")
        with open(summary_path, 'r') as f:
            original_summary = json.load(f)
    except Exception as e:
        print(f"❌ Error fetching summary: {e}")
        return

    final_report = {}
    # Resume from existing file
    if os.path.exists(FINAL_REPORT_FILE):
        with open(FINAL_REPORT_FILE, 'r') as f:
            final_report = json.load(f)

    for i, (model_key, meta_data) in enumerate(original_summary.items(), 1):
        model_name = meta_data.get('nn', model_key)
        pth_filename = f"{model_name}.pth"
        
        # --- Resume Logic ---
        if model_name in final_report:
            prev_data = final_report[model_name]
            prev_status = prev_data.get('status', 'unknown')
            prev_drop = prev_data.get('improvement', {}).get('accuracy_drop', 1.0)
            
            if "success" in prev_status and prev_drop < ACCURACY_DROP_LIMIT:
                print(f"⏭️  Skipping {model_name} (Good Result: Drop {prev_drop:.4f})")
                continue
        
        print(f"-----------------------------------------------------------")
        print(f"🔄 Processing [{i}/{len(original_summary)}]: {model_name}")

        py_path = os.path.join(LOCAL_MODEL_DIR, f"{model_name}.py")
        if not os.path.exists(py_path):
            final_report[model_name] = {"status": "failed", "error_type": "Code missing"}
            continue

        Net = load_model_class(py_path)
        if Net is None: 
            final_report[model_name] = {"status": "failed", "error_type": "Class error"}
            continue

        try:
            # 1. Base Setup (FP32)
            json_prm = meta_data.get('prm', {})
            safe_prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.0}
            if json_prm: safe_prm.update(json_prm)
            
            try:
                model_fp32 = Net(in_shape=(1, 3, 32, 32), out_shape=(10,), prm=safe_prm, device=torch.device("cpu"))
            except:
                raise RuntimeError("Arch Mismatch")

            ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=pth_filename)
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model_fp32.load_state_dict(state_dict, strict=False)

            acc_fp32 = meta_data.get('accuracy', 0.0)
            size_fp32 = get_file_size_mb(ckpt_path)
            
            final_model = None
            final_method = "failed"
            final_acc = 0.0
            final_time = 0.0

            # ======================================================
            # 🟢 PLAN A: Static Quantization (Standard Eager Mode)
            # ======================================================
            try:
                print("   🔵 Trying Plan A (Static)...")
                model_static = copy.deepcopy(model_fp32)
                model_static = QuantWrapper(model_static) # Wrapper lagaya
                model_static.eval()
                model_static.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model_static, inplace=True)
                
                with torch.no_grad():
                    for idx, (inputs, _) in enumerate(calib_loader):
                        if idx >= 1: break
                        model_static(inputs)
                
                torch.quantization.convert(model_static, inplace=True)
                
                # Test Immediately
                acc, time_sec = evaluate_model(model_static, test_loader)
                drop = acc_fp32 - acc
                
                if acc == -1.0: raise ValueError("Static Crashed")
                if drop > ACCURACY_DROP_LIMIT: raise ValueError(f"High Drop: {drop:.2f}")
                
               
                final_model = model_static
                final_method = "success_static"
                final_acc = acc
                final_time = time_sec
                print(f"   ✅ Plan A Passed! Drop: {drop:.4f}")

            except Exception as e:
                print(f"   ⚠️ Plan A Failed ({str(e)[:40]}...). Switching to Plan B...")
                
                # ======================================================
                # 🟡 PLAN B: Dynamic Quantization (The Savior)
                # ======================================================
                try:
                    # Reload Fresh FP32 (Zaroori hai)
                    model_dynamic = copy.deepcopy(model_fp32)
                    
                    # Dynamic Convert
                    model_dynamic = torch.quantization.quantize_dynamic(
                        model_dynamic, 
                        {nn.Linear, nn.Conv2d}, 
                        dtype=torch.qint8
                    )
                    
                    acc, time_sec = evaluate_model(model_dynamic, test_loader)
                    
                    if acc == -1.0: raise ValueError("Dynamic Crashed (Rare)")
                    
                    final_model = model_dynamic
                    final_method = "success_dynamic"
                    final_acc = acc
                    final_time = time_sec
                    print(f"   ✅ Plan B Passed! Drop: {acc_fp32 - acc:.4f}")
                    
                except Exception as e_dyn:
                    print(f"   ❌ Plan B Failed: {e_dyn}")
                    raise e_dyn

            # Save Result
            q_save_path = os.path.join(OUTPUT_FOLDER, f"quantized_{model_name}.pth")
            torch.save(final_model.state_dict(), q_save_path)
            size_int8 = get_file_size_mb(q_save_path)
            
            comparison_data = {
                "status": final_method,
                "metadata": { "dataset": meta_data.get("dataset") },
                "original_fp32": { "accuracy": acc_fp32, "size_mb": round(size_fp32, 2) },
                "quantized_int8": {
                    "accuracy": round(final_acc, 4),
                    "size_mb": round(size_int8, 2),
                    "inference_time_sec": round(final_time, 4)
                },
                "improvement": {
                    "size_reduction_x": round(size_fp32 / size_int8, 2) if size_int8 > 0 else 0,
                    "accuracy_drop": round(acc_fp32 - final_acc, 4)
                }
            }
            final_report[model_name] = comparison_data

        except Exception as e:
            final_report[model_name] = {
                "status": "failed",
                "error_type": "Critical Failure",
                "raw_error": str(e)
            }
        
        # Save JSON on every step
        with open(FINAL_REPORT_FILE, 'w') as f:
            json.dump(final_report, f, indent=4)

if __name__ == "__main__":
    main()
