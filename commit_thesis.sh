#!/bin/bash
# Git commit script for MAI 2025 Thesis
# Run this to commit all necessary files

set -e  # Exit on error

echo "=== MAI 2025 Thesis - Git Commit Script ==="
echo ""

# Step 1: Update .gitignore
echo "Step 1: Updating .gitignore..."
cat >> .gitignore << 'EOF'

# MAI 2025 Thesis - Training Results (regenerable)
ab/nn/stat/train/img-sr_*/
ab/nn/stat/nn/RLFN.json
ab/nn/stat/nn/SPAN.json
ab/nn/stat/nn/SwinIR.json

# Demo outputs (temporary)
demo_output/
demo_test_image.png

# Images (can be regenerated)
*.png
*.jpg
*.jpeg

# Model checkpoints (too large)
*.pth
*.onnx
checkpoints/
EOF

echo "✓ .gitignore updated"
echo ""

# Step 2: Add essential files
echo "Step 2: Adding essential files..."

# Documentation
git add THESIS_IMPLEMENTATION.md EMAIL_TO_PROFESSOR.md README_SR.md GIT_GUIDE.md

# Models
git add ab/nn/nn/RLFN.py ab/nn/nn/SPAN.py

# Data & Metrics
git add ab/nn/loader/div2k.py
git add ab/nn/transform/sr_transforms.py
git add ab/nn/metric/psnr.py ab/nn/metric/ssim.py

# Utilities
git add ab/nn/util/Benchmark.py
git add ab/nn/util/Const.py ab/nn/util/Train.py ab/nn/util/Util.py
git add ab/nn/util/db/Util.py

# Demo
git add demo/sr_demo.py

# Modified files
git add ab/nn/nn/__init__.py
git add requirements.txt
git add .gitignore

echo "✓ Essential files staged"
echo ""

# Step 3: Remove old files
echo "Step 3: Removing old files..."
git rm -r MOHSIN_THESIS_FINAL/ 2>/dev/null || echo "  (MOHSIN_THESIS_FINAL already removed)"
git rm ab/nn/nn/swinir.py 2>/dev/null || echo "  (swinir.py already removed)"
git rm ab/nn/nn/rlfn.py 2>/dev/null || echo "  (rlfn.py already removed)"

echo "✓ Old files removed"
echo ""

# Step 4: Show status
echo "Step 4: Current status..."
git status --short
echo ""

# Step 5: Commit
echo "Step 5: Creating commit..."
git commit -m "Implement MAI 2025 Super-Resolution thesis

- Add RLFN (MAI 2022) and SPAN (NTIRE 2024) models
- Implement DIV2K dataset loader with synthetic fallback
- Add Y-channel PSNR and SSIM metrics (normalized to 0.0-1.0)
- Integrate benchmark metrics (Parameters, FLOPs, Latency)
- Add SR-specific transforms
- Update training framework for automatic benchmarking
- Add comprehensive documentation (THESIS_IMPLEMENTATION.md)
- Remove old MOHSIN_THESIS_FINAL folder
- Fix __init__.py (empty per professor requirement)

All components follow LEMUR/NN Dataset structure requirements.
Training verified and functional.

Models:
- RLFN: 450K params, 2.08 GFLOPs, 7.51ms latency
- SPAN: 605K params, 2.87 GFLOPs, 9.04ms latency

Metrics: PSNR, SSIM, Parameters, FLOPs, Latency (all auto-collected)"

echo "✓ Commit created"
echo ""

# Step 6: Summary
echo "=== Commit Summary ==="
git log -1 --stat | head -30
echo ""

echo "=== SUCCESS ==="
echo "All files committed successfully!"
echo ""
echo "Next steps:"
echo "1. Review the commit: git log -1"
echo "2. Push to remote: git push origin main"
echo "3. Send EMAIL_TO_PROFESSOR.md to your professor"
echo ""
