import os
import sys
import json
import tarfile
import pickle
import urllib.request
import numpy as np
import torch
import warnings
from huggingface_hub import hf_hub_download
import importlib.util

# Warnings chupane k liye
warnings.filterwarnings("ignore")

print(f"🚀 STARTING FINAL GOLD PIPELINE")
print(f"ℹ️  Torch Version: {torch.__version__}")

# ==========================================
# 1. IMPORTS & COMPATIBILITY
# ==========================================
try:
    import ai_edge_torch
    from ai_edge_torch.quantize.quant_config import QuantConfig
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config
    )
except ImportError as e:
    print(f"❌ Import Error: {e}"); sys.exit(1)

# Fix for 'is_dynamic' error
class ConfigWrapper:
    def __init__(self, original_config):
        self.original_config = original_config
        self.is_dynamic = False
        self.is_qat = getattr(original_config, 'is_qat', False)
    def __getattr__(self, name): return getattr(self.original_config, name)

try:
    from ai_edge_litert.interpreter import Interpreter
    print(f"✅ LiteRT Interpreter Loaded")
except ImportError:
    print("❌ LiteRT missing."); sys.exit(1)

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
REPO_ID = "NN-Dataset/checkpoints-epoch-50"       
LOCAL_MODEL_DIR = os.path.join("ab", "nn", "nn") 
OUTPUT_FOLDER = "final_tflite_results_epoch50"
FINAL_REPORT_FILE = "TFLITE_REPORT_EPOCH50.json"
TEST_BATCH_LIMIT = 20  # Test on 20 batches (approx 640 images) to save time
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_DIR = "./data"

# ==========================================
# 🛠️ DATA LOADING
# ==========================================
def get_manual_dataloader(batch_size=32):
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    filepath = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    if not os.path.exists(filepath): urllib.request.urlretrieve(CIFAR_URL, filepath)
    if not os.path.exists(os.path.join(DATA_DIR, "cifar-10-batches-py")):
        with tarfile.open(filepath, "r:gz") as tar: tar.extractall(path=DATA_DIR)

    def load_batch(f):
        with open(f, 'rb') as fo: return pickle.load(fo, encoding='bytes')
    
    batch_path = os.path.join(DATA_DIR, "cifar-10-batches-py", "test_batch")
    data_dict = load_batch(batch_path)
    
    # Pre-process
    images = data_dict[b'data'].reshape((-1, 3, 32, 32)).astype(np.float32) / 255.0
    labels = np.array(data_dict[b'labels'])
    
    for i in range(0, len(images), batch_size):
        yield torch.tensor(images[i:i+batch_size]), torch.tensor(labels[i:i+batch_size])

# ==========================================
# 🧠 SMART VALIDATION LOGIC
# ==========================================
def evaluate_smart(tflite_path, limit_batches):
    # Interpreter Setup
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    in_idx = input_details[0]['index']
    out_idx = output_details[0]['index']
    
    # Model kya chahta hai?
    model_input_shape = input_details[0]['shape'] # e.g. [32, 3, 32, 32]
    req_batch_size = model_input_shape[0]
    nhwc = model_input_shape[-1] == 3 # Check format (N,H,W,C) or (N,C,H,W)
    
    correct = 0; total = 0
    
    # Data Loader usi batch size ka banao jo model ko chahiye
    loader = get_manual_dataloader(batch_size=req_batch_size)
    
    for i, (images, labels) in enumerate(loader):
        if limit_batches and i >= limit_batches: break
        
        current_bs = images.shape[0]
        
        # Format Handling
        if nhwc: images = images.permute(0, 2, 3, 1)
        input_data = images.numpy()
        
        # PADDING LOGIC: Agar data kam hai, to zeroes bhar do
        if current_bs < req_batch_size:
            pad_needed = req_batch_size - current_bs
            # Shape calculate karo
            pad_shape = (pad_needed, *input_data.shape[1:])
            padding = np.zeros(pad_shape, dtype=np.float32)
            input_data = np.concatenate([input_data, padding], axis=0)
            
        # Inference
        interpreter.set_tensor(in_idx, input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(out_idx)
        
        # Validation (Sirf asli data count karo, padding ignore)
        for j in range(current_bs):
            if np.argmax(output[j]) == labels[j].item():
                correct += 1
            total += 1
            
    return correct / total if total > 0 else 0

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 1. Prepare Calibration Batch (High Accuracy Key)
    print("📊 Preparing Calibration Batch (32 Images)...")
    calib_imgs = []
    gen = get_manual_dataloader(1)
    for _ in range(32):
        try: calib_imgs.append(next(gen)[0])
        except: break
    calibration_batch = torch.cat(calib_imgs, dim=0) # [32, 3, 32, 32]

    try:
        summary_path = hf_hub_download(repo_id=REPO_ID, filename="all_models_summary.json")
        with open(summary_path, 'r') as f: original_summary = json.load(f)
    except Exception as e: print(f"❌ Download Error: {e}"); return

    final_report = {}
    if os.path.exists(FINAL_REPORT_FILE):
        try:
            with open(FINAL_REPORT_FILE, 'r') as f: final_report = json.load(f)
        except: pass

    for i, (model_key, meta_data) in enumerate(original_summary.items(), 1):
        model_name = meta_data.get('nn', model_key)
        print(f"--------------------------------------------------")
        print(f"🔄 Processing [{i}/{len(original_summary)}]: {model_name}")

        if model_name in final_report and final_report[model_name].get("status") == "success":
            print(f"   ⏭️  Skipping (Done)")
            continue

        py_path = os.path.join(LOCAL_MODEL_DIR, f"{model_name}.py")
        if not os.path.exists(py_path): continue
        
        try:
            # Setup FP32
            spec = importlib.util.spec_from_file_location("dynamic_model", py_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            Net = module.Net
            safe_prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.0}
            if 'prm' in meta_data: safe_prm.update(meta_data['prm'])
            
            # Setup Model
            model_fp32 = Net(in_shape=(1, 3, 32, 32), out_shape=(10,), prm=safe_prm, device="cpu").eval()
            ckpt = hf_hub_download(repo_id=REPO_ID, filename=f"{model_name}.pth")
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            if "state_dict" in sd: sd = sd["state_dict"]
            model_fp32.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)

            # Quant Config
            quantizer = XNNPACKQuantizer()
            patched_config = ConfigWrapper(get_symmetric_quantization_config())
            quantizer.set_global(patched_config)
            q_config = QuantConfig(pt2e_quantizer=quantizer)

            # 🎯 CONVERT (Using Big Batch for Accuracy)
            # Ye step model ko 32 batch size par lock kar dega
            tflite_model = ai_edge_torch.convert(model_fp32, (calibration_batch,), quant_config=q_config)
            
            # Export
            tflite_path = os.path.join(OUTPUT_FOLDER, f"{model_name}.tflite")
            tflite_model.export(tflite_path)
            size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"      ✅ Saved: {size_mb:.2f} MB")

            # 🎯 VALIDATE (Using Smart Logic)
            acc = evaluate_smart(tflite_path, TEST_BATCH_LIMIT)
            print(f"      🎯 Accuracy: {acc:.4f}")

            final_report[model_name] = {
                "status": "success",
                "tflite_size_mb": round(size_mb, 2),
                "tflite_accuracy": round(acc, 4),
                "original_accuracy": meta_data.get('accuracy', 0.0),
                "drop": round(meta_data.get('accuracy', 0.0) - acc, 4)
            }
            with open(FINAL_REPORT_FILE, 'w') as f: json.dump(final_report, f, indent=4)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            final_report[model_name] = {"status": "failed", "error": str(e)}
            with open(FINAL_REPORT_FILE, 'w') as f: json.dump(final_report, f, indent=4)

if __name__ == "__main__":
    main()
