import torch
import time
import json
import os
import glob
import numpy as np
from ab.nn.nn.rlfn import Net as RLFN
from ab.nn.nn.swinir import Net as SwinIR

# 1. SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📉 Measuring Speed on {device}...")

def get_best_metric(folder):
    # Reads the JSON files to find the best accuracy/score
    files = glob.glob(f"{folder}/*.json")
    if not files: return 0.0
    
    best_val = 0.0
    for f_path in files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                # The framework might save it as 'value', 'accuracy', or 'psnr'
                val = data.get('value', data.get('accuracy', data.get('psnr', 0.0)))
                if val > best_val: best_val = val
        except:
            continue
    return best_val

def measure_speed(model):
    # Measures how fast the model architecture is
    model.to(device).eval()
    dummy = torch.rand(1, 3, 48, 48).to(device) # Standard input size
    times = []
    with torch.no_grad():
        for _ in range(10): model(dummy) # Warmup GPU
        for _ in range(50):
            start = time.time()
            model(dummy)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append(time.time() - start)
    return 1.0 / np.mean(times), sum(p.numel() for p in model.parameters())

# 2. GATHER DATA
print("📊 Reading results from RESULTS_RLFN and RESULTS_SWINIR...")

# RLFN Data
score_r = get_best_metric("RESULTS_RLFN")
fps_r, params_r = measure_speed(RLFN())

# SwinIR Data
score_s = get_best_metric("RESULTS_SWINIR")
fps_s, params_s = measure_speed(SwinIR())

# 3. PRINT TABLE
print("\n" + "="*65)
print(f"{'METRIC':<20} | {'RLFN (Mobile)':<20} | {'SwinIR (Quality)':<20}")
print("-" * 65)
print(f"{'Parameters':<20} | {params_r:<20,d} | {params_s:<20,d}")
print(f"{'Score (Acc/PSNR)':<20} | {score_r:<20.4f} | {score_s:<20.4f}")
print(f"{'Speed (FPS)':<20} | {fps_r:<20.2f} | {fps_s:<20.2f}")
print("="*65 + "\n")
