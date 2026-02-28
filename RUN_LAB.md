# Super-Resolution Training Guide (Lab PC)

This document contains instructions to run the super-resolution models on the Lab GPU PC after fetching the latest changes.

## 1. Prerequisites (Data Setup)
Since the dataset is not pushed to GitHub, ensure that the `DIV2K` dataset is available locally on the Lab PC at `data/DIV2K`.

The directory structure inside `nn-dataset/data/DIV2K/` should look like this:
- `DIV2K_train_HR/` (Directory with high-res training images)
- `DIV2K_train_LR_bicubic/X4/` (Directory with low-res X4 training images)
- `DIV2K_valid_HR/`  (Directory with high-res validation images)
- `DIV2K_valid_LR_bicubic/X4/` (Directory with low-res X4 validation images)

*(Note: Data resolution config is already updated in `ab/nn/loader/div2k.py` to target 720x1280 LR and 2160x3840 HR, assuming scale=3, loaded from the X4 directories).*

## 2. Environment Activation
Activate the python virtual environment before running the training commands:
```bash
source .venv/bin/activate
```
*(If the environment `.venv` does not exist, run the setup scripts provided in the project).*

## 3. Training Commands (10 Epochs Each)
Run the following commands sequentially in the terminal. Each command runs 10 epochs for the specific Model and Metric combination.

### RLFN
```bash
./train.sh -c "img-super-resolution_div2k_psnr_RLFN" -e 10 --save_pth_weights 1
./train.sh -c "img-super-resolution_div2k_ssim_RLFN" -e 10 --save_pth_weights 1
```

### SPAN
```bash
./train.sh -c "img-super-resolution_div2k_psnr_SPAN" -e 10 --save_pth_weights 1
./train.sh -c "img-super-resolution_div2k_ssim_SPAN" -e 10 --save_pth_weights 1
```

### SAFMN
```bash
./train.sh -c "img-super-resolution_div2k_psnr_SAFMN" -e 10 --save_pth_weights 1
./train.sh -c "img-super-resolution_div2k_ssim_SAFMN" -e 10 --save_pth_weights 1
```

### ECBSR
```bash
./train.sh -c "img-super-resolution_div2k_psnr_ECBSR" -e 10 --save_pth_weights 1
./train.sh -c "img-super-resolution_div2k_ssim_ECBSR" -e 10 --save_pth_weights 1
```

## 4. Convert to TFLite (For Android Testing)
Once training is successful, the `.pth` weights will be saved in `out/nn/weights/`. 
To use these on Android, you will need to convert them to `.tflite`. 

Use the provided library `ai_edge_torch` that's already in the `.venv`. Here is an example to convert the generated models:
```python
import torch
import ai_edge_torch

# Example: Load your trained model
from ab.nn.nn.RLFN import Net
model_fp32 = Net(in_shape=(1, 3, 720, 1280), out_shape=(1, 3, 2160, 3840), prm={'lr': 0.01}, device="cpu").eval()

# Load the generated weights
model_fp32.load_state_dict(torch.load('out/nn/weights/img-super-resolution_div2k_psnr_RLFN.pth', map_location="cpu"))

# Create a sample input (1 Batch, 3 Channels, 720x1280 resolution)
sample_input = torch.randn(1, 3, 720, 1280)

# Convert to TFLite
tflite_model = ai_edge_torch.convert(model_fp32, (sample_input,))
tflite_model.export("RLFN_super_resolution.tflite")
```

## 5. Final Submission
Once all 8 commands finish running, the results will be saved in `ab/nn/stat/train/`. 
Add those new statistics to git, commit them, push to the branch, and then open a **Pull Request** to the professor's main repository.
