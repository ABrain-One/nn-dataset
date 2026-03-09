import sys
import os
from ab.nn.util.Const import ab_root_path

# --- AUTO-PATH SETUP (Make script Plug & Play) ---
# Automatically detects project root and adds it to system path.
# This ensures the script runs without manually setting 'export PYTHONPATH'.
try:
    if ab_root_path not in sys.path:
        sys.path.append(ab_root_path)
    print(f'✅ Project Root detected at: {ab_root_path}')
except Exception as e:
    print(f'⚠️ Warning: Auto-path setup failed: {e}')

# --- STANDARD IMPORTS ---
import json
import shutil
import time
from pathlib import Path

# --- PROJECT IMPORTS ---
try:
    from ab.nn.api import data
    from ab.nn.train import main as train_main
    from ab.nn.util.Const import stat_train_dir, ckpt_dir, HF_NN
    from ab.nn.util.Util import release_memory
    import ab.nn.util.hf.HF as HF
    from ab.nn.util.db.Util import unique_nn_cls
except ImportError:
    print("\n❌ Critical Error: Could not import 'ab.nn' modules.")
    print("   Please ensure you are running this script from the project root or 'cmd/py' folder.")
    sys.exit(1)

# --- CONFIGURATION ---
SUMMARY_FILENAME = 'all_models.json'

# =================================================================
# ⚠️ SETTINGS
# =================================================================
TEST_MODE = False
TEST_LIMIT = 10


# =================================================================

def get_existing_models_and_summary(repo_id):
    """
    Downloads the Master JSON from Hugging Face.
    """
    print('☁️ Fetching Master Summary from Hugging Face...')
    if os.path.exists(SUMMARY_FILENAME):
        os.remove(SUMMARY_FILENAME)

    summary_data = {}
    uploaded_models = set()

    try:
        local_path = HF.download(repo_id, SUMMARY_FILENAME, '.')
        with open(local_path, 'r') as f:
            summary_data = json.load(f)

        # Note: We are not deleting the local file so it can be inspected if needed.

        uploaded_models = set(summary_data.keys())
        print(f'✅ Found Master JSON with {len(uploaded_models)} records.')

    except Exception as e:
        print(f'⚠️ Master JSON not found (Starting fresh): {e}')

    return uploaded_models, summary_data


def upload_to_hf(model_name, epoch_max, dataset, task, metric, accuracy, summary_data, repo_id, prm):
    print(f'☁️ Uploading {model_name} to Hugging Face...')

    # --- SUPER FIX: BLIND SEARCH ---
    # Strategy: Since we clean the output folder before training, 
    # we don't need to match the filename. We simply grab the newest .pth file found.
    local_checkpoint = None

    # Check 1: ckpt_dir (standard path)
    files_in_ckpt = list(ckpt_dir.rglob('*.pth'))
    # Check 2: 'out/checkpoints' (Manual fallback)
    files_in_out = list(Path('out/checkpoints').rglob('*.pth'))
    # Check 3: Current directory recursive (Last resort)
    files_in_curr = list(Path('.').rglob('*.pth'))

    # Combine results
    all_found = files_in_ckpt + files_in_out + files_in_curr

    # Filter: Remove .venv files and duplicates
    valid_files = list(set([f for f in all_found if '.venv' not in str(f) and 'site-packages' not in str(f)]))

    if valid_files:
        # Pick the most recent file (Created within the last minute)
        latest_file = max(valid_files, key=os.path.getmtime)
        print(f'   🔍 Super-Search found latest file: {latest_file}')
        local_checkpoint = latest_file
    else:
        # Green warning (No panic) - likely low accuracy model
        print('   ℹ️ No .pth file generated (Likely due to low accuracy). Uploading Metadata only.')

    try:
        HF.upload_file(repo_id, local_checkpoint, f'{model_name}.pth')

        # 2. Update Master Data
        new_metadata = {
            'nn': model_name,
            'accuracy': accuracy,
            'epoch': epoch_max,
            'dataset': dataset,
            'task': task,
            'metric': metric,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'prm': prm
        }
        if new_metadata:
            summary_data[model_name] = new_metadata

            # Save locally (NO DELETE)
            with open(SUMMARY_FILENAME, 'w') as f:
                json.dump(summary_data, f, indent=4)

            # 3. Upload Master JSON
            HF.upload_file(repo_id, SUMMARY_FILENAME, SUMMARY_FILENAME)

        print(f'✅ Successfully processed {model_name}')

        print('⏳ Waiting 30s to avoid Rate Limit...')
        time.sleep(30)

        return True
    except Exception as e:
        print(f'❌ Upload failed: {e}')
        if '429' in str(e):
            print('🛑 Hit Rate Limit! Waiting 2 minutes...')
            time.sleep(120)
        return False


def main():
    try:
        print('📊 Fetching models from API...')
        epoch_max = 5
        epoch_train_max = 50
        dataset = 'cifar-10'
        task = 'img-classification'
        metric = 'acc'
        REPO_NAME = 'checkpoints-epoch-' + str(epoch_train_max)
        repo_id = f'{HF_NN}/{REPO_NAME}'
        df = unique_nn_cls(epoch_max, dataset, task, metric)

        if TEST_MODE:
            df = df[:TEST_LIMIT]
            print(f'⚠️ TEST MODE: Only running first {TEST_LIMIT} models.')

        uploaded_models, summary_data = get_existing_models_and_summary(repo_id)

    except Exception as e:
        print(f'❌ Error initializing: {e}')
        return

    print(f'🔥 Starting Pipeline for {len(df)} models...')

    for i, dt in df.iterrows():
        print(f"\n{'=' * 60}")
        model = dt['nn']
        print(f'Processing Model: {i}/{len(df)} | {model}')

        if model in uploaded_models:
            print(f'⏭️ Skipping {model} (Found in Master JSON)')
            continue

        # Local cleanup per model to ensure fresh state
        if os.path.isdir(ckpt_dir):
            try:
                shutil.rmtree(ckpt_dir)
            except:
                pass

        if os.path.isdir('out/checkpoints'):
            try:
                shutil.rmtree('out/checkpoints')
            except:
                pass

        print(f'🚀 Starting FRESH training for {model}...')
        try:
            params = dt['prm']
            accuracy = train_main(
                config=f'{task}_{dataset}_{metric}_{model}',
                nn_prm=params,
                epoch_max=epoch_train_max,
                n_optuna_trials=-1,
                save_pth_weights=True,
                save_onnx_weights=False,
                train_missing_pipelines=False,
                num_workers=0
            )
            print(f'✅ Training completed call for {model}')
            if accuracy and upload_to_hf(model, epoch_train_max, dataset, task, metric, accuracy, summary_data, repo_id, params):
                uploaded_models.add(model)
        except Exception as e:
            print(f'❌ Training failed/crashed for {model}: {e}')


if __name__ == '__main__':
    main()
