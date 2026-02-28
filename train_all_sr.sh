#!/bin/bash
if [ -d .venv ]; then
    source .venv/bin/activate
fi

# Arrays for models and metrics
models=("RLFN" "SPAN" "SAFMN" "ECBSR")
metrics=("psnr" "ssim")

export PYTHONPATH="."

for model in "${models[@]}"; do
    for metric in "${metrics[@]}"; do
        config="img-super-resolution_div2k_${metric}_${model}"
        echo "================================================================"
        echo "🚀 Starting Training Pipeline for: $config"
        echo "================================================================"
        
        # Train and save .pth weights (10 Epochs)
        ./train.sh -c "$config" -e 10 --save_pth_weights 1
        
        echo "✅ Training done for $model ($metric). Converting .pth to .tflite..."
        
        # Inline Python script to convert generated .pth to .tflite automatically
        python3 -c "
import torch, os, sys
try:
    import ai_edge_torch
except ImportError:
    print('WARNING: ai_edge_torch not found, skipping TFLite conversion for $model')
    sys.exit(0)

# Import the model
from ab.nn.nn.$model import Net
try:
    # 1. Load model structure with 2160x3840 settings
    model_fp32 = Net(in_shape=(1, 3, 720, 1280), out_shape=(1, 3, 2160, 3840), prm={'lr': 0.01}, device='cpu').eval()
    
    # 2. Load the trained weights
    weight_path = 'out/nn/weights/${config}.pth'
    if not os.path.exists(weight_path):
        print(f'ERROR: Weights not found at {weight_path}')
        sys.exit(1)
        
    model_fp32.load_state_dict(torch.load(weight_path, map_location='cpu'))
    
    # 3. Create dummy input & Convert
    sample_input = torch.randn(1, 3, 720, 1280)
    tflite_model = ai_edge_torch.convert(model_fp32, (sample_input,))
    
    # 4. Save TFLite Model
    out_dir = 'out/nn/tflite'
    os.makedirs(out_dir, exist_ok=True)
    tflite_model.export(os.path.join(out_dir, '${config}.tflite'))
    print(f'✅ Successfully converted $model to Android TFLite Format!')
except Exception as e:
    print(f'❌ Failed to convert to TFLite: {e}')
"
    done
done

echo "================================================================"
echo "🎉 ALL DONE! Stats, .pth weights, and .tflite files are ready! 🎉"
echo "================================================================"
