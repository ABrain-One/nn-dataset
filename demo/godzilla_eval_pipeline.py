import torch
import math
import os
import numpy as np
from PIL import Image

def to_tensor(pil_img):
    # Convert PIL Image to [C, H, W] float tensor [0, 1]
    arr = np.array(pil_img).transpose(2, 0, 1) # [H, W, C] -> [C, H, W]
    return torch.from_numpy(arr).float() / 255.0

def center_crop(pil_img, size):
    # PIL Center Crop (equivalent to torchvision center_crop)
    w, h = pil_img.size
    th, tw = size
    x1 = (w - tw) // 2
    y1 = (h - th) // 2
    return pil_img.crop((x1, y1, x1 + tw, y1 + th))

WEIGHT_PATH = "out/ckpt/HSRv4/best_model.pth"
DATA_ROOT   = "data/DIV2K"
SCALE       = 3
LR_CROP     = (720, 1280)   # input size
HR_CROP     = (2160, 3840)  # output size

def psnr_db(pred, target):
    pred   = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    from ab.nn.nn.HSRv4 import Net
    model = Net(in_shape=(3, 720, 1280), out_shape=(3, 2160, 3840), prm={}, device=device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded weights from: {WEIGHT_PATH}")

    # Validation set
    hr_dir = os.path.join(DATA_ROOT, "DIV2K_valid_HR")
    lr_dir = os.path.join(DATA_ROOT, f"DIV2K_valid_LR_bicubic/X{SCALE}")

    files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])
    psnr_list = []

    print(f"\n🚀 Running Super Resolution Test:")
    print(f"   Input Size:  {LR_CROP[1]}x{LR_CROP[0]} (720p)")
    print(f"   Output Size: {HR_CROP[1]}x{HR_CROP[0]} (4K)")
    print(f"   Evaluating {len(files)} images...\n")

    with torch.no_grad():
        for i, fname in enumerate(files):
            hr_path = os.path.join(hr_dir, fname)
            lr_name = fname.replace('.png', f'x{SCALE}.png')
            lr_path = os.path.join(lr_dir, lr_name)

            if not os.path.exists(lr_path):
                continue

            lr_pil = center_crop(Image.open(lr_path).convert('RGB'), LR_CROP)
            hr_pil = center_crop(Image.open(hr_path).convert('RGB'), HR_CROP)
            
            lr_img = to_tensor(lr_pil).unsqueeze(0).to(device)
            hr_img = to_tensor(hr_pil).unsqueeze(0).to(device)

            try:
                sr_img = model(lr_img)
                p = psnr_db(sr_img, hr_img)
                psnr_list.append(p)
                print(f"  [{i+1:3d}/{len(files)}] {fname}: {p:.2f} dB")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [{i+1:3d}] OOM on {fname}, skipping...")
                    torch.cuda.empty_cache()
                else:
                    raise e

    if psnr_list:
        avg = sum(psnr_list) / len(psnr_list)
        print(f"\n{'='*50}")
        print(f"Average PSNR @ {HR_CROP[1]}x{HR_CROP[0]} (4K): {avg:.2f} dB")
        print(f"Score in competition format (÷48): {avg/48:.4f}")
        print(f"Images evaluated: {len(psnr_list)}/{len(files)}")
        print(f"Competition requirement: > 30 dB  → {'✅ PASS' if avg > 30 else '❌ FAIL'}")
    else:
        print("No images evaluated!")

if __name__ == "__main__":
    main()
