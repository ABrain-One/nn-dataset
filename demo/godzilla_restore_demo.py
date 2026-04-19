import torch
import os
from ab.nn.nn.HSRv4 import HSR_Godzilla

# This script restores the Godzilla v4 model from a checkpoint and 
# serves as a template for TFLite conversion as required by the challenge.

def restore_and_convert():
    print("🚀 Restoring Godzilla v4 (180 Channels)...")
    
    # Init Model
    model = HSR_Godzilla(feature_channels=180, n_groups=5, n_blocks=8, upscale=3)
    
    # Load Checkpoint
    checkpoint_path = "best_model.pth"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict, strict=False)
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}. Ensure it is in the same directory.")
        return

    model.eval()
    
    # Note: TFLite conversion logic using onnx_converter.py is recommended 
    # to maintain the static 1x720x1280x3 NHWC input format required for NPU stability.
    print("✅ Model is ready for inference/conversion.")

if __name__ == "__main__":
    restore_and_convert()
