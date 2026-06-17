"""
TitanV5 Conversion Pipeline: PyTorch → ONNX → TF SavedModel → TFLite
"""
import torch
import os
import sys
import subprocess
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ab.nn.nn.TitanV5 import TitanV7_Lite_Custom

def export_onnx():
    DEVICE = 'cpu'
    ONNX_PATH = "TitanV5.onnx"

    print(f"🚀 Step 1: Exporting TitanV7_Lite_Custom (f=32) to ONNX...")
    
    # Initialize the new f=32 model
    model = TitanV7_Lite_Custom(f=32, n_blocks=3, upscale=3, groups=4).to(DEVICE)
    
    print("🚀 Loading actual trained weights...")
    state_dict = torch.load("out/ckpt/TitanV5/best_model.pth", map_location=DEVICE)
    
    # Strip 'model.' prefix if necessary
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        
    model.load_state_dict(state_dict, strict=False)
    print("✅ Weights loaded successfully!")
    
    # Merge training branches before exporting
    print("🪄 Reparameterizing / Deploying RepConv branches...")
    model.deploy()
    model.eval()

    # Add torch.clamp wrapper for output normalization
    class WrappedTitanV5(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return torch.clamp(self.m(x), 0.0, 1.0)
            
    wrapped_model = WrappedTitanV5(model)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Loaded and deployed dummy weights | Params: {params:.2f}M")

    dummy_input = torch.randn(1, 3, 720, 1280)

    torch.onnx.export(
        wrapped_model, dummy_input, ONNX_PATH,
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes=None
    )

    size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
    print(f"✨ ONNX exported: {ONNX_PATH} ({size_mb:.2f} MB)")
    return ONNX_PATH

def convert_onnx_to_tf(onnx_path):
    tf_dir = "TitanV5_TF"
    print(f"\n🚀 Step 2: ONNX → TF SavedModel...")
    cmd = ["onnx2tf", "-i", onnx_path, "-o", tf_dir, "-n"]
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"✅ TF SavedModel generated in {tf_dir}/")
        return tf_dir
    else:
        print("❌ onnx2tf CLI failed")
        return None

def convert_to_tflite(tf_dir):
    import tensorflow as tf
    from PIL import Image

    print(f"\n🚀 Step 3: TF SavedModel → TFLite...")
    out_dir = "out/nn/tflite/titan_v5"
    os.makedirs(out_dir, exist_ok=True)

    def do_convert(mode):
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)

        if mode == "fp32":
            out_name = f"{out_dir}/model_none_float.tflite"
            print("   💎 FP32...")

        elif mode == "int8_flex":
            out_name = f"{out_dir}/model_none.tflite"
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("   🎛️ INT8 Flexible...")

        elif mode == "int8_fixed":
            out_name = f"{out_dir}/model.tflite"
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            def representative_data_gen():
                lr_dir = "data/DIV2K/DIV2K_valid_LR_bicubic/X3"
                if os.path.exists(lr_dir):
                    files = sorted(os.listdir(lr_dir))[:5]
                    for fname in files:
                        img = Image.open(os.path.join(lr_dir, fname)).convert("RGB")
                        img = img.resize((1280, 720), Image.BICUBIC)
                        img_np = np.array(img).astype(np.float32) / 255.0
                        img_np = img_np[np.newaxis, ...]
                        yield [img_np]
                else:
                    print("   ⚠️ No calibration images found. Using random data.")
                    for _ in range(3):
                        yield [np.random.randn(1, 720, 1280, 3).astype(np.float32)]

            converter.representative_dataset = representative_data_gen
            print("   ⚡ INT8 Fixed (NPU/GPU)...")

        try:
            tflite_model = converter.convert()
            with open(out_name, "wb") as f:
                f.write(tflite_model)
            size_mb = os.path.getsize(out_name) / 1024 / 1024
            print(f"   ✅ {out_name} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"   ❌ {mode} failed: {e}")

    do_convert("fp32")
    do_convert("int8_flex")
    do_convert("int8_fixed")

if __name__ == "__main__":
    print("=" * 60)
    print("🏗️ TitanV5: PyTorch → ONNX → TF → TFLite")
    print("=" * 60)

    onnx_path = export_onnx()
    tf_dir = convert_onnx_to_tf(onnx_path)
    if tf_dir:
        convert_to_tflite(tf_dir)

    print("\n" + "=" * 60)
    print("🏆 TitanV5 conversion complete! Files: out/nn/tflite/titan_v5/")
    print("=" * 60)
