# HSRv4 (Godzilla) - Super Resolution Model

## Overview

**HSRv4** (nicknamed "Godzilla") is a high-performance single-image super-resolution model designed for 3x upscaling (720p → 4K). It achieves **33.77 dB PSNR** on the DIV2K validation set.

| Metric | Value |
|--------|-------|
| PSNR (FP32) | 33.77 dB |
| Parameters | ~4.2M |
| Scale | 3x (720p → 4K) |
| Dataset | DIV2K |

---

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision pillow numpy
# For TFLite conversion:
pip install onnx onnx2tf tensorflow
```

### 2. Prepare Dataset

Download DIV2K dataset and place it at:
```
nn-dataset/data/DIV2K/
├── DIV2K_train_HR/
├── DIV2K_valid_HR/
└── DIV2K_valid_LR_bicubic/X3/
```

### 3. Train HSRv4

Run via the native framework:
```bash
cd nn-dataset
./train.sh -c "img-super-resolution_div2kv2_psnr_HSRv4" -e 400 -f "sr_transforms"
```

Trained weights will be saved to:
```
out/ckpt/HSRv4/best_model.pth
```

---

## Evaluation

```bash
PYTHONPATH=. python3 -c "
import torch
from ab.nn.nn.HSRv4 import Net
from ab.nn.loader.div2kv2 import Div2KDataset

model = Net(3, 3)
state = torch.load('out/ckpt/HSRv4/best_model.pth', map_location='cpu')
model.load_state_dict({k.replace('model.',''):v for k,v in state.items()})
model.eval()
print('Model loaded! Params:', sum(p.numel() for p in model.parameters())/1e6, 'M')
"
```

---

## TFLite Conversion

See [`convert_HSRv4_tflite.py`](convert_HSRv4_tflite.py) in this folder for the complete PyTorch → ONNX → TF → TFLite pipeline.

```bash
cd nn-dataset
PYTHONPATH=. python3 demo/convert_HSRv4_tflite.py
```

Output files will be saved to `out/submission/HSRv4/TFLite/`:
- `model.tflite` — INT8 Fixed `[1, 720, 1280, 3]`
- `model_none.tflite` — INT8 Dynamic `[1, None, None, 3]`
- `model_none_float.tflite` — FP32 Dynamic `[1, None, None, 3]`

---

## Architecture

HSRv4 uses a deep residual architecture with:
- **180 feature channels** for maximum capacity
- **Enhanced Spatial Attention (ESA)** blocks
- **PixelShuffle** upsampling (3x)
- **Global Residual Learning** (bicubic base + CNN residual)
- Knowledge Distillation teacher for downstream student models (TitanV4, TitanV5)
