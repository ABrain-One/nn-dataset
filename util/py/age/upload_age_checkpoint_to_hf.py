#!/usr/bin/env python3
"""
Simplified Age Estimation Checkpoint Upload to HuggingFace
This script uploads the best age estimation checkpoint to HuggingFace for mobile pipeline testing.
"""

import os
import json
import time
from pathlib import Path
from huggingface_hub import HfApi, login, hf_hub_download
import argparse

# Configuration
REPO_ID = "Arun03k/checkpoints-epoch-50"
CHECKPOINT_PATH = "cluster_artifacts/final_training_age_estimation/final_training_age_estimation.pth"
SUMMARY_FILENAME = "all_models_summary.json"

# Model metadata based on our training results
MODEL_METADATA = {
    "MobileAgeNet": {
        "nn": "MobileAgeNet",
        "accuracy": 4.7372,  # Best MAE from epoch 52
        "epoch": 50,
        "dataset": "utkface",
        "task": "age-regression",
        "metric": "mae",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "prm": {
            "lr": 0.0013274116387530662,
            "batch": 64,
            "dropout": 0.22840538092707335,
            "backbone_lr_mult": 0.1,
            "transform": "Resize_ColorJit_Flip_Blur"
        },
        "pth_uploaded": True
    }
}

def upload_checkpoint(model_name="MobileAgeNet", dry_run=False):
    """
    Upload age estimation checkpoint to HuggingFace
    """
    print(f"🚀 Starting Age Estimation Checkpoint Upload")
    print(f"📁 Checkpoint: {CHECKPOINT_PATH}")
    print(f"☁️ Repository: {REPO_ID}")
    print(f"🤖 Model: {model_name}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint file not found: {CHECKPOINT_PATH}")
        return False

    checkpoint_size = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
    print(f"📊 Checkpoint size: {checkpoint_size:.2f} MB")

    if dry_run:
        print("🧪 DRY RUN - Would upload but skipping actual upload")
        return True

    try:
        # Initialize HF API
        api = HfApi()

        # Download existing summary or create new one
        summary_data = {}
        try:
            print("📥 Downloading existing summary from HuggingFace...")
            local_summary = hf_hub_download(repo_id=REPO_ID, filename=SUMMARY_FILENAME)
            with open(local_summary, 'r') as f:
                summary_data = json.load(f)
            print(f"✅ Found existing summary with {len(summary_data)} models")
        except Exception as e:
            print(f"ℹ️ No existing summary found, creating new one: {e}")

        # Upload checkpoint file
        print(f"📤 Uploading checkpoint: {model_name}.pth...")
        api.upload_file(
            path_or_fileobj=CHECKPOINT_PATH,
            path_in_repo=f"{model_name}.pth",
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("✅ Checkpoint upload complete")

        # Update summary
        summary_data[model_name] = MODEL_METADATA[model_name]

        # Save and upload updated summary
        with open(SUMMARY_FILENAME, 'w') as f:
            json.dump(summary_data, f, indent=4)

        print("📤 Uploading updated summary...")
        api.upload_file(
            path_or_fileobj=SUMMARY_FILENAME,
            path_in_repo=SUMMARY_FILENAME,
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("✅ Summary upload complete")

        # Clean up local summary
        if os.path.exists(SUMMARY_FILENAME):
            os.remove(SUMMARY_FILENAME)

        print(f"🎉 Successfully uploaded {model_name} to HuggingFace!")
        print(f"🔗 View at: https://huggingface.co/{REPO_ID}/tree/main")

        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload age estimation checkpoint to HuggingFace')
    parser.add_argument('--model-name', default='MobileAgeNet', help='Model name to upload')
    parser.add_argument('--dry-run', action='store_true', help='Run without actually uploading')
    args = parser.parse_args()

    print("🔐 Please make sure you're logged into HuggingFace:")
    print("   huggingface-cli login")
    print("   or set HUGGINGFACE_TOKEN environment variable")
    print()

    if upload_checkpoint(args.model_name, args.dry_run):
        print("✅ Upload process completed successfully")
    else:
        print("❌ Upload process failed")

if __name__ == "__main__":
    main()