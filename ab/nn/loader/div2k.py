
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import ab.nn.transform.sr_transforms as sr_transforms

scale = 4

class Div2KDataset(Dataset):
    def __init__(self, root, mode='train', scale=4, transform=None):
        self.mode = mode
        self.scale = scale
        self.transform = transform
        self.use_synthetic = False
        
        # Paths for real DIV2K dataset
        if mode == 'train':
            self.hr_dir = os.path.join(root, 'DIV2K_train_HR')
            self.lr_dir = os.path.join(root, f'DIV2K_train_LR_bicubic/X{scale}')
        else:
            self.hr_dir = os.path.join(root, 'DIV2K_valid_HR')
            self.lr_dir = os.path.join(root, f'DIV2K_valid_LR_bicubic/X{scale}')
        
        # Check if real dataset exists
        if os.path.exists(self.hr_dir) and os.path.exists(self.lr_dir):
            print(f"✓ Using real DIV2K dataset from {root}")
            self.hr_files = sorted([os.path.join(self.hr_dir, x) for x in os.listdir(self.hr_dir) if x.endswith('.png')])
            self.lr_files = sorted([os.path.join(self.lr_dir, x) for x in os.listdir(self.lr_dir) if x.endswith('.png')])
            self.use_synthetic = False
        else:
            print(f"⚠ DIV2K dataset not found at {root}")
            print(f"⚠ Generating synthetic images for testing...")
            self.use_synthetic = True
            # Generate 100 synthetic images for train, 10 for test
            self.num_samples = 100 if mode == 'train' else 10

    def __len__(self):
        if self.use_synthetic:
            return self.num_samples
        return len(self.hr_files)

    def __getitem__(self, idx):
        if self.use_synthetic:
            # Generate synthetic HR and LR images
            hr_size = 192
            lr_size = hr_size // self.scale  # 48x48
            
            # Create a random colored image with some pattern
            np.random.seed(idx + (0 if self.mode == 'train' else 10000))
            
            # HR image: random colored gradient
            hr_array = np.random.randint(0, 256, (hr_size, hr_size, 3), dtype=np.uint8)
            hr_img = Image.fromarray(hr_array, mode='RGB')
            
            # LR image: downsampled version
            lr_img = hr_img.resize((lr_size, lr_size), Image.BICUBIC)
            
        else:
            # Load real images
            hr_img = Image.open(self.hr_files[idx]).convert('RGB')
            lr_img = Image.open(self.lr_files[idx]).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            lr_img, hr_img = self.transform(lr_img, hr_img)
        
        # Convert to tensors
        lr_tensor = F.to_tensor(lr_img)
        hr_tensor = F.to_tensor(hr_img)
        
        return lr_tensor, hr_tensor

def loader(transform, task):
    """
    Load DIV2K dataset for super resolution.
    Falls back to synthetic data if DIV2K is not downloaded.
    
    Args:
        transform: Transform function/object (IGNORED - SR needs special transforms)
        task: Task name (e.g., 'img-sr')
    
    Returns:
        out_shape: Output shape tuple (C, H, W)
        minimum_accuracy: Minimum acceptable PSNR (0.0 for SR)
        train_set: Training dataset
        test_set: Test dataset
    """
    root = 'data/DIV2K'
    scale = 4  # Default upscaling factor
    
    # ALWAYS use SR-specific transforms (ignore passed transform which may be for classification)
    # SR transforms need to operate on BOTH LR and HR images simultaneously
    train_transform = sr_transforms.Compose([
        sr_transforms.RandomHorizontalFlip(),
        sr_transforms.RandomVerticalFlip(),
        sr_transforms.RandomRotation()
    ])
    
    train_set = Div2KDataset(root, mode='train', scale=scale, transform=train_transform)
    test_set = Div2KDataset(root, mode='test', scale=scale, transform=None)
    
    # Calculate output shape based on scale factor
    # For DIV2K with scale 4: LR is 48x48, HR is 192x192
    hr_size = 192
    out_shape = (3, hr_size, hr_size)
    
    # Minimum accuracy: 0.0 means any positive PSNR is considered valid progress
    minimum_accuracy = 0.0
    
    return out_shape, minimum_accuracy, train_set, test_set
