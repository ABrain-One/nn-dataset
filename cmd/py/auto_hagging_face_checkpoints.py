import json
import os
import shutil
import time

from huggingface_hub import HfApi

from ab.nn.api import data
from ab.nn.train import main as train_main
from ab.nn.util.Const import stat_train_dir, ckpt_dir
from ab.nn.util.Util import release_memory

# --- CONFIGURATION ---
# Token should be provided via environment variable or passed securely
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = "NN-Dataset"
REPO_NAME = "checkpoints"

# =================================================================
# ⚠️ SETTINGS
# =================================================================
# Agar tumhein SAARE models (hazaaron) par chalana hai, 
# to isay False kar do.
TEST_MODE = False
TEST_LIMIT = 10
# =================================================================

# --- AUTOMATIC MODEL SELECTION ---
try:
    print("📊 Fetching all available models from API...")
    df = data()
    all_models_list = df['nn'].unique().tolist()

    # FILTER: Sirf Clean Models uthao
    MODELS_TO_TRAIN = [
        m for m in all_models_list
        if "Bayesian" not in m and "GAN" not in m and "LSTM" not in m
           and "RNN" not in m and "C10C" not in m and "C5C" not in m
           and "C8C" not in m and "Transformer" not in m
    ]

    print(f"✅ Found {len(MODELS_TO_TRAIN)} clean models in total.")

    if TEST_MODE:
        MODELS_TO_TRAIN = MODELS_TO_TRAIN[:TEST_LIMIT]
        print(f"⚠️ TESTING MODE ON: Processing only first {TEST_LIMIT} models.")
    else:
        print(f"🚀 PRODUCTION MODE: READY TO OVERWRITE ALL {len(MODELS_TO_TRAIN)} MODELS.")

except Exception as e:
    print(f"❌ Error fetching model list: {e}")
    MODELS_TO_TRAIN = ["AlexNet"]


def get_best_params(model_name):
    print(f"🔍 Finding best parameters for {model_name}...")
    try:
        df = data(nn=model_name)
        if df.empty: return None

        # Best params select karo
        if 'acc' in df.columns:
            best_row = df.sort_values(by='acc', ascending=False).iloc[0]
        elif 'accuracy' in df.columns:
            best_row = df.sort_values(by='accuracy', ascending=False).iloc[0]
        else:
            best_row = df.sort_values(by='duration', ascending=True).iloc[0]

        params = best_row['prm']

        # SAFETY: Force Batch Size 32 (OOM se bachne ke liye)
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


def train_and_save(model_name, params):
    print(f"🚀 Starting FRESH training for {model_name}...")
    config_pattern = f"img-classification_cifar-10_acc_{model_name}"

    try:
        release_memory()

        # Training call
        train_main(
            config=config_pattern,
            nn_prm=params,
            epoch_max=1,  # Cluster par isay barha dena
            n_optuna_trials=-1,  # Force New Training (Overwrite logic)
            save_pth_weights=True,
            save_onnx_weights=False,
            train_missing_pipelines=False,
            num_workers=0  # Memory Crash fix
        )
        print(f"✅ Training completed call for {model_name}")
        return True
    except Exception as e:
        print(f"❌ Training failed/crashed for {model_name}: {e}")
        return False


def create_metadata_file(model_name):
    """Generates metadata.json from training stats."""
    try:
        config_name = f"img-classification_cifar-10_acc_{model_name}"
        stat_path = stat_train_dir / config_name

        if not stat_path.exists(): return None

        json_files = list(stat_path.glob("*.json"))
        if not json_files: return None

        latest_json = max(json_files, key=os.path.getmtime)

        with open(latest_json, 'r') as f:
            data = json.load(f)

        # Handle List vs Dict format
        stats = {}
        if isinstance(data, list):
            if data: stats = data[-1]
        elif isinstance(data, dict):
            stats = data

        if not stats: return None

        metadata = {
            "model_name": model_name,
            "accuracy": stats.get('accuracy', 0),
            "epoch": stats.get('epoch', 0),
            "duration_sec": stats.get('duration', 0) / 1e9,
            "batch_size": stats.get('prm', {}).get('batch', 32),
            "dataset": "cifar-10",
            "task": "img-classification",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        metadata_path = f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        return metadata_path

    except Exception as e:
        print(f"⚠️ Failed to create metadata: {e}")
        return None


def upload_to_hf(model_name):
    print(f"☁️ Uploading {model_name} to Hugging Face...")
    checkpoint_dir = ckpt_dir / model_name
    expected_file = checkpoint_dir / 'best_model.pth'

    # 1. Generate Metadata
    metadata_file = create_metadata_file(model_name)

    api = HfApi(token=HF_TOKEN)
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        # 2. Upload .pth (Overwrite if exists)
        api.upload_file(
            path_or_fileobj=str(expected_file),
            path_in_repo=f"{model_name}/{model_name}.pth",
            repo_id=repo_id,
            repo_type="model"
        )

        # 3. Upload JSON (Overwrite if exists)
        if metadata_file:
            api.upload_file(
                path_or_fileobj=metadata_file,
                path_in_repo=f"{model_name}/metadata.json",
                repo_id=repo_id,
                repo_type="model"
            )
            os.remove(metadata_file)  # Cleanup local file

        print(f"✅ Successfully uploaded {model_name} (Weights + Metadata)")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def main():
    total_models = len(MODELS_TO_TRAIN)
    print(f"🔥 Starting FULL OVERWRITE Pipeline for {total_models} models...")

    for i, model in enumerate(MODELS_TO_TRAIN, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing Model: {i}/{total_models} | {model}")
        print(f"{'=' * 60}")

        # NOTE: Humne "Skip Logic" hata di hai.
        # Ye ab pehle se maujood models ko bhi dobara train karega.

        params = get_best_params(model)
        if not params:
            print("Skipping (No Params found)")
            continue
        if os.path.isdir(ckpt_dir):
            try:
                shutil.rmtree(ckpt_dir)  # # Cleanup all local checkpoint files
                print(f"Directory {ckpt_dir} cleaned.")
            except Exception as e:
                print(f"Error removing directory {ckpt_dir}: {e}")

        success = train_and_save(model, params)
        if success:
            upload_to_hf(model)

        release_memory()


if __name__ == "__main__":
    main()
