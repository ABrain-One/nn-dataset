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
import copy
import json
import shutil
import time
from pathlib import Path
import argparse

# --- PROJECT IMPORTS (core only; train/HF loaded when needed) ---
try:
    from ab.nn.util.Const import ckpt_dir, HF_NN, core_nn_cls
except ImportError as exc:
    print("\n❌ Critical Error: Could not import 'ab.nn' modules.")
    print(f"   Details: {exc}")
    print("   Try: export PYTHONPATH=/path/to/nn-dataset:$PYTHONPATH")
    sys.exit(1)


def load_train_main():
    from ab.nn.train import main as train_main
    return train_main


def load_hf():
    import ab.nn.util.hf.HF as HF
    return HF

# --- CONFIGURATION ---
SUMMARY_FILENAME = 'all_models.json'

# =================================================================
# ⚠️ SETTINGS
# =================================================================
TEST_MODE = False
TEST_LIMIT = 10

REQUIRED_PRM_KEYS = ('batch', 'lr', 'momentum', 'transform')

# Fallback hyperparameters when the statistics DB has no entry for a model/dataset pair.
# Each dataset keeps its own defaults (ImageNet-style vs CIFAR-style).
DATASET_DEFAULT_PRM = {
    'cifar-10': {
        'batch': 128,
        'lr': 0.01,
        'momentum': 0.9,
        'transform': 'complex',
    },
    'cifar-100': {
        'batch': 128,
        'lr': 0.01,
        'momentum': 0.9,
        'transform': 'complex',
    },
    'imagenet100': {
        'batch': 256,
        'lr': 0.01,
        'momentum': 0.9,
        'transform': 'norm_160_flip',
    },
    'imagenette': {
        'batch': 256,
        'lr': 0.01,
        'momentum': 0.9,
        'transform': 'norm_160_flip',
    },
}


# Models whose code expects complex-valued inputs (torch.complex64).
COMPLEX_DATASET_TRANSFORM = {
    'cifar-10': 'complex',
    'cifar-100': 'complex',
    'imagenet100': 'complex_160_flip',
    'imagenette': 'complex_160_flip',
}

_COMPLEX_NN_CACHE = {}


# =================================================================

def is_complex_nn(nn):
    """Detect architectures that require complex input tensors."""
    if nn in _COMPLEX_NN_CACHE:
        return _COMPLEX_NN_CACHE[nn]
    nn_file = Path(ab_root_path) / 'ab' / 'nn' / 'nn' / f'{nn}.py'
    if not nn_file.is_file():
        _COMPLEX_NN_CACHE[nn] = False
        return False
    text = nn_file.read_text(encoding='utf-8', errors='ignore')
    needs_complex = 'apply_complex' in text and 'input.imag' in text
    _COMPLEX_NN_CACHE[nn] = needs_complex
    return needs_complex


def dataset_default_prm(dataset):
    if dataset not in DATASET_DEFAULT_PRM:
        raise ValueError(
            f"No default training parameters for dataset '{dataset}'. "
            f"Supported datasets: {', '.join(sorted(DATASET_DEFAULT_PRM))}"
        )
    return copy.deepcopy(DATASET_DEFAULT_PRM[dataset])


def default_prm_for(nn, dataset):
    """Per-model dataset defaults (complex architectures get complex transforms)."""
    prm = dataset_default_prm(dataset)
    if is_complex_nn(nn):
        prm['transform'] = COMPLEX_DATASET_TRANSFORM.get(dataset, prm['transform'])
    return prm


def merge_prm(prm_from_db, dataset, nn):
    prm = default_prm_for(nn, dataset)
    if isinstance(prm_from_db, dict):
        for key, value in prm_from_db.items():
            if value is not None and value == value:
                prm[key] = value
    if is_complex_nn(nn) and not str(prm.get('transform', '')).startswith('complex'):
        prm['transform'] = COMPLEX_DATASET_TRANSFORM.get(dataset, 'complex')
    return prm


def prm_is_complete(prm):
    if not isinstance(prm, dict):
        return False
    return all(key in prm and prm[key] is not None and prm[key] == prm[key] for key in REQUIRED_PRM_KEYS)


def fetch_db_stats(epoch_max, dataset, task, metric):
    """Load DB stats; return empty frame when no rows exist for this dataset."""
    import pandas as pd
    from ab.nn.api import data

    df = pd.concat([
        data(nn_prefixes=('rag-', 'unq-'), only_best_accuracy=True, task=task, dataset=dataset, metric=metric, epoch=epoch_max),
        data(nn=core_nn_cls, only_best_accuracy=True, task=task, dataset=dataset, metric=metric, epoch=epoch_max),
    ], ignore_index=True)
    if df.empty or 'accuracy' not in df.columns:
        return pd.DataFrame()
    return df.sort_values(by='accuracy', ascending=False)


def build_training_plan(epoch_max, dataset, task, metric, models=None):
    """
    Build one training job per core classifier.

    Uses the best DB statistics when available; otherwise falls back to dataset defaults.
    """
    models = list(models or core_nn_cls)
    df_stats = fetch_db_stats(epoch_max, dataset, task, metric)

    stats_by_nn = {}
    if not df_stats.empty:
        for _, row in df_stats.iterrows():
            nn = row['nn']
            if nn not in stats_by_nn:
                stats_by_nn[nn] = row

    plan = []
    db_count = 0
    fallback_count = 0

    for nn in models:
        if nn in stats_by_nn:
            row = stats_by_nn[nn]
            prm = merge_prm(row.get('prm'), dataset, nn)
            source = 'db' if prm_is_complete(row.get('prm')) else 'db+default'
            if is_complex_nn(nn):
                source = f'{source}+complex-transform'
            db_count += 1
            plan.append({
                'nn': nn,
                'prm': prm,
                'param_source': source,
                'db_accuracy': row.get('accuracy'),
            })
        else:
            fallback_count += 1
            plan.append({
                'nn': nn,
                'prm': default_prm_for(nn, dataset),
                'param_source': 'default+complex-transform' if is_complex_nn(nn) else 'default',
                'db_accuracy': None,
            })

    return plan, db_count, fallback_count


def resolve_repo_name(dataset, epoch_train_max, repo_name=None):
    if repo_name:
        return repo_name
    base = f'checkpoints-epoch-{epoch_train_max}'
    if dataset == 'cifar-10':
        return base
    return f'{base}-{dataset}'


def get_existing_models_and_summary(repo_id):
    """
    Downloads the Master JSON from Hugging Face.
    """
    print('☁️ Fetching Master Summary from Hugging Face...')
    if os.path.exists(SUMMARY_FILENAME):
        os.remove(SUMMARY_FILENAME)

    summary_data = {}
    uploaded_models = set()
    HF = load_hf()

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


def upload_to_hf(model_name, epoch_max, dataset, task, metric, accuracy, summary_data, repo_id, prm, param_source):
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
        # Upload checkpoint file only if found
        if local_checkpoint:
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
            'prm': prm,
            'param_source': param_source,
            'pth_uploaded': bool(local_checkpoint)
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
        parser = argparse.ArgumentParser(description='Upload checkpoints to Hugging Face')
        parser.add_argument('--task', default='img-classification', help='Task name (e.g. age-regression)')
        parser.add_argument('--dataset', default='cifar-10', help='Dataset name (e.g. utkface)')
        parser.add_argument('--metric', default='acc', help='Metric name (e.g. mae)')
        parser.add_argument('--epoch-train-max', type=int, default=50, help='Epochs used for final training')
        parser.add_argument('--model', help='Train only this core model (e.g. ComplexNet)')
        parser.add_argument('--repo-name', help='HF repo name (default: checkpoints-epoch-N[-dataset])')
        parser.add_argument('--test-mode', action='store_true', help='Run in test mode (limited models)')
        parser.add_argument('--dry-run', action='store_true', help='Print training plan and exit')
        args = parser.parse_args()

        print('📊 Building training plan...')
        epoch_max = 5
        epoch_train_max = args.epoch_train_max
        dataset = args.dataset
        task = args.task
        metric = args.metric
        test_mode = args.test_mode
        models = (args.model,) if args.model else None
        plan, db_count, fallback_count = build_training_plan(epoch_max, dataset, task, metric, models=models)

        if test_mode:
            plan = plan[:TEST_LIMIT]
            print(f'⚠️ TEST MODE: Only running first {TEST_LIMIT} models.')

        repo_name = resolve_repo_name(dataset, epoch_train_max, args.repo_name)
        repo_id = f'{HF_NN}/{repo_name}'

        print(f'📦 HF target repo: {repo_id}')
        print(f'🔥 Training plan: {len(plan)} models (db={db_count}, fallback={fallback_count})')
        for job in plan:
            print(f"   - {job['nn']}: {job['param_source']} -> {job['prm']}")

        if args.dry_run:
            print('✅ Dry run complete (no training started).')
            return

        uploaded_models, summary_data = get_existing_models_and_summary(repo_id)

    except Exception as e:
        print(f'❌ Error initializing: {e}')
        return

    print(f'🔥 Starting Pipeline for {len(plan)} models...')
    print(f'   DB params: {db_count} | Default fallback: {fallback_count}')

    for i, job in enumerate(plan, start=1):
        print(f"\n{'=' * 60}")
        model = job['nn']
        params = job['prm']
        param_source = job['param_source']
        print(f'Processing Model: {i}/{len(plan)} | {model} | params={param_source}')
        if param_source != 'db':
            print(f'   Using fallback prm: {params}')

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
            train_main = load_train_main()
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
            if accuracy and upload_to_hf(
                model, epoch_train_max, dataset, task, metric, accuracy,
                summary_data, repo_id, params, param_source,
            ):
                uploaded_models.add(model)
        except Exception as e:
            print(f'❌ Training failed/crashed for {model}: {e}')


if __name__ == '__main__':
    main()
