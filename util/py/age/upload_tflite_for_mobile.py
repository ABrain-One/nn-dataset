#!/usr/bin/env python3
"""
Upload TFLite Age Estimation Models to HuggingFace for Mobile Testing
This uploads the generated TFLite models to trigger the mobile device testing pipeline.
"""

import os
import json
import time
from pathlib import Path
from huggingface_hub import HfApi, login, upload_file, upload_folder
import argparse

# Configuration
TFLITE_REPO_ID = "Arun03k/tflite"
LOCAL_TFLITE_DIR = "ab/nn/stat/run/age-regression_utkface_mae_MobileAgeNet"
REPORT_FILE = "ab/nn/imp/TFLITE_REPORT_AGE_EPOCH50.json"

def create_all_models_json():
    """Create all_models.json for the mobile testing pipeline"""

    # Read our TFLite report
    with open(REPORT_FILE, 'r') as f:
        tflite_report = json.load(f)

    # Create all_models.json in expected format
    all_models = {}
    for model_name, report in tflite_report.items():
        all_models[model_name] = {
            "name": model_name,
            "task": "age-regression",
            "dataset": "utkface",
            "metric": "mae",
            "accuracy": report.get("pytorch_mae", 9.09),  # Use PyTorch MAE
            "tflite_size_mb": report.get("tflite_size_mb", 0.11),
            "uploaded": True,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }

    return all_models

def upload_tflite_models():
    """Upload TFLite models and metadata to HuggingFace"""

    print(f"🚀 Starting TFLite Upload to Mobile Testing Pipeline")
    print(f"📁 Local TFLite Dir: {LOCAL_TFLITE_DIR}")
    print(f"☁️  Repository: {TFLITE_REPO_ID}")

    if not os.path.exists(LOCAL_TFLITE_DIR):
        print(f"❌ TFLite directory not found: {LOCAL_TFLITE_DIR}")
        return False

    if not os.path.exists(REPORT_FILE):
        print(f"❌ TFLite report not found: {REPORT_FILE}")
        return False

    try:
        api = HfApi()

        # Create all_models.json
        all_models = create_all_models_json()
        all_models_path = "all_models.json"
        with open(all_models_path, 'w') as f:
            json.dump(all_models, f, indent=4)

        # Upload directory structure: int8/age-regression_utkface_mae_MobileAgeNet/
        remote_path = "int8/age-regression_utkface_mae_MobileAgeNet/"

        print(f"📤 Uploading TFLite models...")

        # Upload all .tflite files in the directory
        for filename in os.listdir(LOCAL_TFLITE_DIR):
            if filename.endswith('.tflite'):
                local_file = os.path.join(LOCAL_TFLITE_DIR, filename)
                remote_file = f"{remote_path}{filename}"

                print(f"  📤 Uploading {filename}...")
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=remote_file,
                    repo_id=TFLITE_REPO_ID,
                    repo_type="model"
                )

                size_mb = os.path.getsize(local_file) / (1024 * 1024)
                print(f"     ✅ {filename}: {size_mb:.2f} MB")

        # Upload all_models.json to trigger pipeline
        print(f"📤 Uploading all_models.json...")
        api.upload_file(
            path_or_fileobj=all_models_path,
            path_in_repo=f"int8/age-regression_utkface_mae/all_models.json",
            repo_id=TFLITE_REPO_ID,
            repo_type="model"
        )

        # Clean up
        if os.path.exists(all_models_path):
            os.remove(all_models_path)

        print(f"🎉 Successfully uploaded TFLite models!")
        print(f"🔗 View at: https://huggingface.co/{TFLITE_REPO_ID}/tree/main")
        print(f"📱 Mobile testing should start automatically...")
        print(f"🔍 Monitor results at: https://huggingface.co/{TFLITE_REPO_ID}/tree/main/int8/age-regression_utkface_mae")

        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload TFLite models to HuggingFace for mobile testing')
    args = parser.parse_args()

    print("🔐 Using HuggingFace authentication...")

    if upload_tflite_models():
        print("✅ TFLite upload completed successfully")
        print("📱 Mobile device testing should begin shortly")
        print("🕐 Results typically appear within 15-30 minutes")
    else:
        print("❌ TFLite upload failed")

if __name__ == "__main__":
    main()