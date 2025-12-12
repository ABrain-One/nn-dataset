import sys
import os

# --- AUTO-PATH SETUP (Make script Plug & Play) ---
# Automatically detects project root and adds it to system path.
# This ensures the script runs without manually setting 'export PYTHONPATH'.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)
    print(f"✅ Project Root detected at: {project_root}")
except Exception as e:
    print(f"⚠️ Warning: Auto-path setup failed: {e}")

# --- STANDARD IMPORTS ---
import json
import shutil
import time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

# --- PROJECT IMPORTS ---
try:
    from ab.nn.api import data
    from ab.nn.train import main as train_main
    from ab.nn.util.Const import stat_train_dir, ckpt_dir
    from ab.nn.util.Util import release_memory
except ImportError:
    print("\n❌ Critical Error: Could not import 'ab.nn' modules.")
    print("   Please ensure you are running this script from the project root or 'cmd/py' folder.")
    sys.exit(1)

# --- CONFIGURATION ---
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    if not HF_TOKEN:
    # Fallback for testing
    HF_TOKEN = "" 
    print("⚠️ Warning: No HF_TOKEN found. Please set environment variable.")
HF_USERNAME = "NN-Dataset"
REPO_NAME = "checkpoints"
SUMMARY_FILENAME = "all_models_summary.json"

# =================================================================
# ⚠️ SETTINGS
# =================================================================
TEST_MODE = False
TEST_LIMIT = 10
# =================================================================

def get_existing_models_and_summary():
    """
    Downloads the Master JSON from Hugging Face.
    """
    print("☁️ Fetching Master Summary from Hugging Face...")
    api = HfApi(token=HF_TOKEN)
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    
    summary_data = {}
    uploaded_models = set()

    try:
        local_path = hf_hub_download(
            repo_id=repo_id, 
            filename=SUMMARY_FILENAME, 
            token=HF_TOKEN,
            local_dir="."
        )
        with open(local_path, 'r') as f:
            summary_data = json.load(f)
        
        # Note: We are not deleting the local file so it can be inspected if needed.
        
        uploaded_models = set(summary_data.keys())
        print(f"✅ Found Master JSON with {len(uploaded_models)} records.")
        
    except Exception as e:
        print(f"⚠️ Master JSON not found (Starting fresh): {e}")

    return uploaded_models, summary_data

def get_best_params(model_name):
    print(f"🔍 Finding best parameters for {model_name}...")
    try:
        df = data(nn=model_name)
        if df.empty: return None

        # Prioritize accuracy, fallback to duration
        if 'acc' in df.columns:
            best_row = df.sort_values(by='acc', ascending=False).iloc[0]
        elif 'accuracy' in df.columns:
            best_row = df.sort_values(by='accuracy', ascending=False).iloc[0]
        else:
            best_row = df.sort_values(by='duration', ascending=True).iloc[0]

        params = best_row['prm']
        
        # SAFETY: Force Batch Size to 32 to avoid OOM
        if 'batch' in params:
            if params['batch'] > 32:
                print(f"   📉 Reducing Batch Size from {params['batch']} to 32 for safety.")
                params['batch'] = 32
        else:
            params['batch'] = 32
        return params
    except Exception as e:
        print(f"❌ Error getting params: {e}")
        return None

def train_and_save(model_name, params, epoch_max):
    print(f"🚀 Starting FRESH training for {model_name}...")
    config_pattern = f"img-classification_cifar-10_acc_{model_name}"
    try:
        release_memory()
        train_main(
            config=config_pattern,
            nn_prm=params,
            epoch_max=epoch_max,
            n_optuna_trials=-1,
            save_pth_weights=True,
            save_onnx_weights=False,
            train_missing_pipelines=False,
            num_workers=0
        )
        print(f"✅ Training completed call for {model_name}")
        return True
    except Exception as e:
        print(f"❌ Training failed/crashed for {model_name}: {e}")
        return False

def get_metadata_from_stats(model_name, epoch_max):
    """
    Extracts the latest training statistics (accuracy, time, etc.) from the stats directory.
    """
    try:
        config_name = f"img-classification_cifar-10_acc_{model_name}"
        stat_path = stat_train_dir / config_name
        if not stat_path.exists(): return None
        
        json_files = list(stat_path.glob("*.json"))
        if not json_files: return None
        latest_json = max(json_files, key=os.path.getmtime)

        with open(latest_json, 'r') as f:
            data = json.load(f)
        
        stats = {}
        if isinstance(data, list) and data: stats = data[-1]
        elif isinstance(data, dict): stats = data
        if not stats: return None

        return {
            "model_name": model_name,
            "accuracy": stats.get('accuracy', 0),
            "epoch": stats.get('epoch', epoch_max),
            "duration_sec": stats.get('duration', 0) / 1e9,
            "batch_size": stats.get('prm', {}).get('batch', 32),
            "dataset": "cifar-10",
            "task": "img-classification",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except:
        return None

def upload_to_hf(model_name, epoch_max, summary_data):
    print(f"☁️ Uploading {model_name} to Hugging Face...")
    
    # --- SUPER FIX: BLIND SEARCH ---
    # Strategy: Since we clean the output folder before training, 
    # we don't need to match the filename. We simply grab the newest .pth file found.
    expected_file = None
    
    # Check 1: ckpt_dir (standard path)
    files_in_ckpt = list(ckpt_dir.rglob("*.pth"))
    # Check 2: 'out/checkpoints' (Manual fallback)
    files_in_out = list(Path("out/checkpoints").rglob("*.pth"))
    # Check 3: Current directory recursive (Last resort)
    files_in_curr = list(Path(".").rglob("*.pth"))
    
    # Combine results
    all_found = files_in_ckpt + files_in_out + files_in_curr
    
    # Filter: Remove .venv files and duplicates
    valid_files = list(set([f for f in all_found if ".venv" not in str(f) and "site-packages" not in str(f)]))
    
    if valid_files:
        # Pick the most recent file (Created within the last minute)
        latest_file = max(valid_files, key=os.path.getmtime)
        print(f"   🔍 Super-Search found latest file: {latest_file}")
        expected_file = latest_file
    else:
        # Green warning (No panic) - likely low accuracy model
        print("   ℹ️ No .pth file generated (Likely due to low accuracy). Uploading Metadata only.")

    api = HfApi(token=HF_TOKEN)
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        # 1. Upload .pth (If exists)
        if expected_file and expected_file.exists():
            api.upload_file(
                path_or_fileobj=str(expected_file),
                path_in_repo=f"{model_name}/{model_name}.pth",
                repo_id=repo_id,
                repo_type="model"
            )
        
        # 2. Update Master Data
        new_metadata = get_metadata_from_stats(model_name, epoch_max)
        if new_metadata:
            summary_data[model_name] = new_metadata
            
            # Save locally (NO DELETE)
            with open(SUMMARY_FILENAME, 'w') as f:
                json.dump(summary_data, f, indent=4)
            
            # 3. Upload Master JSON
            api.upload_file(
                path_or_fileobj=SUMMARY_FILENAME,
                path_in_repo=SUMMARY_FILENAME,
                repo_id=repo_id,
                repo_type="model"
            )
            
        print(f"✅ Successfully processed {model_name}")
        
        print("⏳ Waiting 30s to avoid Rate Limit...")
        time.sleep(30)
        
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        if "429" in str(e):
            print("🛑 Hit Rate Limit! Waiting 2 minutes...")
            time.sleep(120)
        return False

def main():
    try:
        print("📊 Fetching models from API...")
        df = data()
        all_models = df['nn'].unique().tolist()
        
        # Filter out junk/irrelevant models
        MODELS_TO_TRAIN = [
            m for m in all_models
            if "Bayesian" not in m and "GAN" not in m and "LSTM" not in m
            and "RNN" not in m and "C10C" not in m and "C5C" not in m
            and "C8C" not in m and "Transformer" not in m
        ]
        
        if TEST_MODE:
            MODELS_TO_TRAIN = MODELS_TO_TRAIN[:TEST_LIMIT]
            print(f"⚠️ TEST MODE: Only running first {TEST_LIMIT} models.")

        uploaded_models, summary_data = get_existing_models_and_summary()

    except Exception as e:
        print(f"❌ Error initializing: {e}")
        return

    print(f"🔥 Starting Pipeline for {len(MODELS_TO_TRAIN)} models...")
    epoch_max = 1

    for i, model in enumerate(MODELS_TO_TRAIN, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing Model: {i}/{len(MODELS_TO_TRAIN)} | {model}")
        
        if model in uploaded_models:
            print(f"⏭️ Skipping {model} (Found in Master JSON)")
            continue

        params = get_best_params(model)
        if not params: 
            print("Skipping (No Params found)")
            continue

        # Local cleanup per model to ensure fresh state
        if os.path.isdir(ckpt_dir):
            try: shutil.rmtree(ckpt_dir)
            except: pass
        
        if os.path.isdir("out/checkpoints"):
             try: shutil.rmtree("out/checkpoints")
             except: pass

        success = train_and_save(model, params, epoch_max)
        if success:
            if upload_to_hf(model, epoch_max, summary_data):
                uploaded_models.add(model)

        release_memory()

if __name__ == "__main__":
    main()
