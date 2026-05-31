#!/usr/bin/env python3
"""
Monitor HuggingFace Mobile Testing Progress
Check for new mobile device results every few minutes
"""

import time
from huggingface_hub import login, HfApi
import subprocess

# Your credentials - set via environment variable or login
TOKEN = None  # Use huggingface_hub login() instead of hardcoded token
REPO_ID = "Arun03k/tflite"

def check_mobile_results():
    """Check for mobile testing results"""
    try:
        # Use environment variable or interactive login
        login()  # This will use stored token or prompt for login
        api = HfApi()
        files = list(api.list_repo_files(REPO_ID, repo_type='model'))

        # Look for android result files
        android_files = [f for f in files if 'android_' in f and '.json' in f]
        age_android_files = [f for f in android_files if 'age-regression' in f]

        print(f"🔍 Checking {REPO_ID} at {time.strftime('%H:%M:%S')}")
        print(f"📁 Total files in repo: {len(files)}")

        if age_android_files:
            print(f"🎉 MOBILE RESULTS FOUND! ({len(age_android_files)} devices)")
            for result in sorted(age_android_files):
                device_name = result.split('/')[-1].replace('android_', '').replace('.json', '')
                print(f"   📱 {device_name}")
            return True
        else:
            print(f"⏱️  No mobile results yet. Looking for files in:")
            expected_path = "int8/age-regression_utkface_mae/"
            matching_path_files = [f for f in files if expected_path in f]
            if matching_path_files:
                print(f"   ✅ Path exists: {expected_path}")
                for f in matching_path_files:
                    print(f"      {f}")
            else:
                print(f"   ⚠️  Expected path not found: {expected_path}")
            return False

    except Exception as e:
        print(f"❌ Error checking repo: {e}")
        return False

def main():
    """Monitor for results with periodic checks"""
    print("🚀 MOBILE TESTING PROGRESS MONITOR")
    print("=" * 50)
    print(f"📱 Monitoring: {REPO_ID}")
    print(f"🔍 Looking for: android_*.json files")
    print(f"⏰ Check every 2 minutes (Ctrl+C to stop)")
    print()

    login(token=TOKEN)

    check_count = 0
    while True:
        check_count += 1
        print(f"\n--- CHECK #{check_count} ---")

        if check_mobile_results():
            print("\n🎉 SUCCESS! Mobile testing results are available!")
            print("🔗 View results at:")
            print(f"   https://huggingface.co/{REPO_ID}/tree/main")
            break

        if check_count == 1:
            print("\n💡 What to expect:")
            print("   - HuggingFace runs your model on real Android devices")
            print("   - Results typically appear in 15-30 minutes")
            print("   - Files will be named like: android_SM-F926B.json")
            print("   - Each file contains inference times for CPU/GPU/NPU")

        try:
            print(f"\n⏱️  Waiting 2 minutes for next check...")
            time.sleep(120)  # Wait 2 minutes
        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped by user")
            break

if __name__ == "__main__":
    main()