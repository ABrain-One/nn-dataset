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

# 👇 Zaroori Imports FX Mode k liye
from torch.ao.quantization import quantize_fx
from torch.ao.quantization import get_default_qconfig_mapping

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
TEST_BATCH_LIMIT = 50 

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
    # FX Mode ko calibration k liye thora ziada data chahiye hota hai
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
                # FX Quantized models kabhi kabhi tuple return karte hain
                if isinstance(outputs, tuple): outputs = outputs[0]
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except Exception:
                return -1.0, 0.0
            
    end_time = time.time()
    inference_time = end_time - start_time
    accuracy = correct / total if total > 0 else 0
    return accuracy, inference_time

def get_file_size_mb(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print(f"\n🚀 STARTING FX STATIC QUANTIZATION (High Success Rate)")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    calib_loader, test_loader = get_dataloaders()
    
    # Dummy Input for Tracing (Zaroori hai FX k liye)
    # CIFAR-10 shape: Batch=1, Channels=3, Height=32, Width=32
    example_input = torch.randn(1, 3, 32, 32)

    try:
        summary_path = hf_hub_download(repo_id=REPO_ID, filename="all_models_summary.json")
        with open(summary_path, 'r') as f:
            original_summary = json.load(f)
    except Exception as e:
        print(f"❌ Error fetching summary: {e}")
        return

    final_report = {}
    if os.path.exists(FINAL_REPORT_FILE):
        with open(FINAL_REPORT_FILE, 'r') as f:
            final_report = json.load(f)

    for i, (model_key, meta_data) in enumerate(original_summary.items(), 1):
        model_name = meta_data.get('nn', model_key)
        pth_filename = f"{model_name}.pth"
        
        print(f"===========================================================")
        print(f"📊 PROGRESS:      [ {i} / {len(original_summary)} ]")
        print(f"👤 MODEL:         {model_name}")

        # Skip logic: 
        if model_name in final_report:
            status = final_report[model_name].get('status', 'unknown')
            if "success_static_fx" in status:
                print(f"⏭️  SKIPPING (Already FX Quantized)")
                continue
        
        py_path = os.path.join(LOCAL_MODEL_DIR, f"{model_name}.py")
        if not os.path.exists(py_path): 
            final_report[model_name] = {"status": "failed", "error_type": "Code missing"}
            continue

        Net = load_model_class(py_path)
        if Net is None: 
            final_report[model_name] = {"status": "failed", "error_type": "Class error"}
            continue

        try:
            # 1. Setup FP32
            json_prm = meta_data.get('prm', {})
            safe_prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.0}
            if json_prm: safe_prm.update(json_prm)
            
            try:
                model_fp32 = Net(in_shape=(1, 3, 32, 32), out_shape=(10,), prm=safe_prm, device=torch.device("cpu"))
                model_fp32.eval() 
            except:
                raise RuntimeError("Arch Mismatch")

            # 2. Load Weights
            print("   📥 Loading weights...")
            try:
                ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=pth_filename)
                state_dict = torch.load(ckpt_path, map_location="cpu")
                if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model_fp32.load_state_dict(state_dict, strict=False)
            except Exception as e:
                # if file not found on HF
                final_report[model_name] = {"status": "failed", "error_type": "Weights Missing on HF"}
                print(f"   ❌ Failed: Weights not found")
                continue

            acc_fp32 = meta_data.get('accuracy', 0.0)
            size_fp32 = get_file_size_mb(ckpt_path)
            
            # ====================================================
            # 🪄 MAGIC: FX GRAPH MODE STATIC QUANTIZATION
            # ====================================================
            print("   🪄 Applying FX Graph Mode (Auto-Fix Add/Cat)...")
            
            # 1. Configuration (FBGEMM for x86 CPUs)
            qconfig_mapping = get_default_qconfig_mapping("fbgemm")
            
            # 2. Prepare (Trace graph & Insert Observers)
            # Ye step 'cat' aur 'add' ko khud detect karke handle karega
            model_prepared = quantize_fx.prepare_fx(model_fp32, qconfig_mapping, example_input)
            
            # 3. Calibrate
            with torch.no_grad():
                for idx, (inputs, _) in enumerate(calib_loader):
                    if idx >= 2: break # 2 batches for better stats
                    model_prepared(inputs)
            
            # 4. Convert
            model_quantized = quantize_fx.convert_fx(model_prepared)
            
            # 5. Test
            print(f"   🧪 Testing Static Accuracy...")
            acc_int8, time_int8 = evaluate_model(model_quantized, test_loader)
            
            if acc_int8 == -1.0:
                raise RuntimeError("Model crashed during inference")

            # Save
            q_save_path = os.path.join(OUTPUT_FOLDER, f"quantized_{model_name}.pth")
            torch.save(model_quantized.state_dict(), q_save_path)
            size_int8 = get_file_size_mb(q_save_path)

            comparison_data = {
                "status": "success_static_fx",
                "metadata": { "dataset": meta_data.get("dataset") },
                "original_fp32": { "accuracy": acc_fp32, "size_mb": round(size_fp32, 2) },
                "quantized_int8": {
                    "accuracy": round(acc_int8, 4),
                    "size_mb": round(size_int8, 2),
                    "inference_time_sec": round(time_int8, 4)
                },
                "improvement": {
                    "size_reduction_x": round(size_fp32 / size_int8, 2) if size_int8 > 0 else 0,
                    "accuracy_drop": round(acc_fp32 - acc_int8, 4)
                }
            }
            final_report[model_name] = comparison_data
            print(f"   ✅ Success! Drop: {comparison_data['improvement']['accuracy_drop']:.4f}")

        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}...")
            final_report[model_name] = {
                "status": "failed",
                "error_type": "FX Tracing Error",
                "raw_error": str(e)
            }
        
        with open(FINAL_REPORT_FILE, 'w') as f:
            json.dump(final_report, f, indent=4)

if __name__ == "__main__":
    main()
