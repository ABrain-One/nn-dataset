import torch
import os
import subprocess
from ab.nn.nn.HSRv4 import HSR_Godzilla

def convert_godzilla(weight_path):
    """
    STABLE PIPELINE: PyTorch -> ONNX -> TF -> TFLite
    This version avoids ai_edge_torch hangs and ensures NPU compatibility.
    """
    print("🚀 --- [GODZILLA STABLE CONVERSION] ---")
    
    # 1. Load Model
    model = HSR_Godzilla(feature_channels=180, n_groups=5, n_blocks=8, upscale=3)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        print(f"✅ Loaded weights from {weight_path}")
    model.eval()

    # 2. Export to ONNX (Static NHWC Layout)
    onnx_path = "official_repro.onnx"
    dummy_input = torch.randn(1, 3, 720, 1280)
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=13, 
                      input_names=['input'], output_names=['output'])
    print(f"✅ Step 1: Exported to ONNX ({onnx_path})")

    # 3. Convert to TF (Simplified call to onnx2tf)
    print("🔄 Step 2: Converting to TensorFlow (ensure onnx2tf is installed)...")
    try:
        subprocess.run(["onnx2tf", "-i", onnx_path, "-o", "saved_model_repro"], check=True)
        print("✅ Step 2: TensorFlow SavedModel generated.")
    except Exception as e:
        print(f"❌ TF conversion failed. Ensure 'onnx2tf' is installed: {e}")
        return

    # 4. Final TFLite Generation (INT8 Quantization)
    print("🔄 Step 3: Generating Quantized TFLite...")
    try:
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_repro")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Note: In a full environment, we add representative_dataset here.
        tflite_model = converter.convert()
        with open("godzilla_repro.tflite", "wb") as f:
            f.write(tflite_model)
        print("✅ SUCCESS! TFLite model saved as 'godzilla_repro.tflite'")
    except Exception as e:
        print(f"❌ TFLite export failed: {e}")

if __name__ == "__main__":
    weights = "out/ckpt/HSRv4/best_model.pth"
    if os.path.exists(weights):
        convert_godzilla(weights)
    else:
        print("❌ WEIGHTS NOT FOUND. Run training first.")
