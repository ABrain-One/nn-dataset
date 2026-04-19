import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class Div2KDataset(Dataset):
    def __init__(self, root, mode='train', scale=3):
        self.mode = mode
        self.scale = scale
        
        sub = 'train' if mode == 'train' else 'valid'
        self.hr_dir = os.path.join(root, f'DIV2K_{sub}_HR')
        self.lr_dir = os.path.join(root, f'DIV2K_{sub}_LR_bicubic/X{scale}')
        
        print(f"--- [LOADER] Initializing {mode} from {self.hr_dir}")
        
        # Get all HR files
        all_hr = sorted([f for f in os.listdir(self.hr_dir) if f.endswith('.png')])
        self.samples = []

        for f in all_hr:
            hr_path = os.path.join(self.hr_dir, f)
            # Apply the naming convention found: [number]x3.png
            lr_name = f.replace('.png', f'x{scale}.png')
            lr_path = os.path.join(self.lr_dir, lr_name)
            
            if os.path.exists(lr_path):
                self.samples.append((lr_path, hr_path))

        print(f"✅ [LOADER] {mode} initialized: {len(self.samples)} pairs found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lr_img = Image.open(self.samples[idx][0]).convert('RGB')
        hr_img = Image.open(self.samples[idx][1]).convert('RGB')
        
        # Patching to 128x128 for training efficiency
        if self.mode == 'train':
            w, h = lr_img.size
            tw, th = 128, 128
            x1 = np.random.randint(0, w - tw)
            y1 = np.random.randint(0, h - th)
            lr_img = lr_img.crop((x1, y1, x1 + tw, y1 + th))
            hr_img = hr_img.crop((x1*self.scale, y1*self.scale, (x1+tw)*self.scale, (y1+th)*self.scale))
        else:
            # Use 256x256 during training eval to avoid GPU freeze
            # Full 720x1280->2160x3840 eval is done separately at competition time
            lr_img = F.center_crop(lr_img, (256, 256))
            hr_img = F.center_crop(hr_img, (256*self.scale, 256*self.scale))

        return F.to_tensor(lr_img), F.to_tensor(hr_img)

def loader(transform, task):
    root = 'data/DIV2K'
    scale = 3
    train_set = Div2KDataset(root, mode='train', scale=scale)
    test_set = Div2KDataset(root, mode='valid', scale=scale)
    
    # (Channel, Height, Width) - using the HR target size
    out_shape = (3, 2160, 3840) 
    return out_shape, 0.0, train_set, test_set
