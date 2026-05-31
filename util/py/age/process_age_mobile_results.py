#!/usr/bin/env python3
"""
Age Estimation Mobile Results Database Integration
This script processes mobile device test results and adds them to the database.
"""

import os
import json
import sqlite3
from pathlib import Path
import argparse

def process_mobile_results(results_dir="ab/nn/stat/run", model_pattern="age-regression_utkface_mae_*"):
    """
    Process all mobile device results for age estimation models
    """
    print(f"🔍 Processing mobile results from: {results_dir}")

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    # Find all age regression model directories
    age_model_dirs = list(results_path.glob(model_pattern))
    print(f"📁 Found {len(age_model_dirs)} age regression model directories")

    total_processed = 0
    for model_dir in age_model_dirs:
        model_name = model_dir.name
        print(f"\n🔄 Processing model: {model_name}")

        # Find all Android result files
        android_files = list(model_dir.glob("android_*.json"))
        print(f"   📱 Found {len(android_files)} device results")

        for android_file in android_files:
            try:
                with open(android_file, 'r') as f:
                    result = json.load(f)

                device_name = result.get('device_type', 'Unknown Device')
                duration = result.get('duration', 0)

                print(f"   ✅ {device_name}: {duration/1000:.1f}ms")

                # Here you would integrate with your database
                # integrate_to_database(result)
                total_processed += 1

            except Exception as e:
                print(f"   ❌ Error processing {android_file}: {e}")

    print(f"\n🎉 Processed {total_processed} mobile device results")
    return total_processed

def generate_summary_report(results_dir="ab/nn/stat/run"):
    """
    Generate a summary report comparing age estimation performance across devices
    """
    print("📊 Generating Performance Summary Report")

    summary = {
        "age_estimation": {},
        "comparison_with_image_classification": {}
    }

    # Process age estimation results
    results_path = Path(results_dir)
    age_model_dirs = list(results_path.glob("age-regression_utkface_mae_*"))

    for model_dir in age_model_dirs:
        model_name = model_dir.name.replace("age-regression_utkface_mae_", "")
        summary["age_estimation"][model_name] = {}

        android_files = list(model_dir.glob("android_*.json"))
        for android_file in android_files:
            try:
                with open(android_file, 'r') as f:
                    result = json.load(f)

                device = result.get('device_type', 'Unknown')
                summary["age_estimation"][model_name][device] = {
                    "inference_time_ms": result.get('duration', 0) / 1000,
                    "cpu_time_ms": result.get('cpu_duration', 0) / 1000,
                    "gpu_time_ms": result.get('gpu_duration', 0) / 1000,
                    "input_shape": f"({result.get('in_dim_1', 'Unknown')}x{result.get('in_dim_2', 'Unknown')})",
                    "task": "age_regression",
                    "metric": "MAE"
                }
            except Exception as e:
                print(f"   ❌ Error processing {android_file}: {e}")

    # Save summary report
    with open("age_estimation_mobile_performance_report.json", 'w') as f:
        json.dump(summary, f, indent=4)

    print("✅ Summary report saved: age_estimation_mobile_performance_report.json")
    return summary

def main():
    parser = argparse.ArgumentParser(description='Process age estimation mobile results')
    parser.add_argument('--results-dir', default='ab/nn/stat/run', help='Results directory')
    parser.add_argument('--summary-only', action='store_true', help='Only generate summary report')
    args = parser.parse_args()

    if args.summary_only:
        generate_summary_report(args.results_dir)
    else:
        process_mobile_results(args.results_dir)
        generate_summary_report(args.results_dir)

if __name__ == "__main__":
    main()