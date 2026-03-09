#!/bin/bash

# 1. Prepare Clean Folders
rm -rf RESULTS_RLFN RESULTS_SWINIR
mkdir -p RESULTS_RLFN
mkdir -p RESULTS_SWINIR

echo "🚀 STARTING FINAL 5-EPOCH RUN (SEPARATED)"
echo "----------------------------------------"

# 2. RLFN Run
echo "➡️  Step 1: Training RLFN..."
python3 -m ab.nn.train -c "super-resolution_div2k_psnr_rlfn" -e 5 --save_pth_weights True
# Capture RLFN JSONs immediately into their own folder
find ab/nn/stat/train -name "*.json" -mmin -5 -exec cp {} RESULTS_RLFN/ \;

# 3. SwinIR Run
echo "➡️  Step 2: Training SwinIR..."
# (Sleep 1s to ensure file timestamps are distinct)
sleep 1
python3 -m ab.nn.train -c "super-resolution_div2k_psnr_swinir" -e 5 --save_pth_weights True
# Capture SwinIR JSONs immediately (exclude the RLFN ones by checking path if needed, 
# but -mmin -1 coupled with the sleep usually works. We'll be safer with path check)
find ab/nn/stat/train -path "*swinir*" -name "*.json" -mmin -5 -exec cp {} RESULTS_SWINIR/ \;

echo "----------------------------------------"
echo "✅ FILES CAPTURED SEPARATELY:"
echo "📂 RLFN Folder:"
ls RESULTS_RLFN | head -3
echo "..."
echo "📂 SwinIR Folder:"
ls RESULTS_SWINIR | head -3
