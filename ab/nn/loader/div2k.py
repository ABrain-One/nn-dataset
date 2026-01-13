import os
import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class Div2KDataset(Dataset):
    def __init__(self, root, mode='train', scale=4):
        self.mode = mode
        self.scale = scale
        
        # 1. Define Paths
        self.hr_dir = os.path.join(root, 'DIV2K_valid_HR')
        base_lr = os.path.join(root, 'DIV2K_valid_LR_bicubic')
        if os.path.exists(os.path.join(base_lr, 'X4')):
            self.lr_dir = os.path.join(base_lr, 'X4')
        else:
            self.lr_dir = base_lr

        # 2. Get File Lists
        self.hr_files = sorted(glob.glob(os.path.join(self.hr_dir, '*.png')))
        self.lr_files = sorted(glob.glob(os.path.join(self.lr_dir, '*.png')))

        if len(self.hr_files) == 0:
            raise RuntimeError(f"No images found in {self.hr_dir}")
        
        # 3. Split: Use first 90 for Train, last 10 for Test
        if mode == 'train':
            self.hr_files = self.hr_files[:90]
            self.lr_files = self.lr_files[:90]
        else:
            self.hr_files = self.hr_files[90:]
            self.lr_files = self.lr_files[90:]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert('RGB')
        lr = Image.open(self.lr_files[idx]).convert('RGB')

        # FIX: We must crop BOTH Train and Test sets so they fit in a batch.
        lr_patch_size = 32
        hr_patch_size = lr_patch_size * self.scale
        
        w, h = lr.size
        
        if self.mode == 'train':
            # Random Crop for Training
            x = random.randint(0, w - lr_patch_size)
            y = random.randint(0, h - lr_patch_size)
        else:
            # Center Crop for Testing (Deterministic)
            x = (w - lr_patch_size) // 2
            y = (h - lr_patch_size) // 2

        # Apply Crop
        lr = lr.crop((x, y, x + lr_patch_size, y + lr_patch_size))
        hr = hr.crop((x * self.scale, y * self.scale, 
                      (x + lr_patch_size) * self.scale, 
                      (y + lr_patch_size) * self.scale))

        return self.to_tensor(lr), self.to_tensor(hr)

def loader(task, dataset_name, transform_name=None):
    root = 'data/DIV2K'
    train_set = Div2KDataset(root, mode='train')
    test_set = Div2KDataset(root, mode='test')
    
    # Return shape must match the crop size (32x4 = 128)
    return (3, 128, 128), 0.0, train_set, test_set
