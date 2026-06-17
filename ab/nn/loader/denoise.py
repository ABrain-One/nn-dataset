import os
import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image
import urllib.request 
import zipfile 
import io

DATASET_ROOT_TRAIN = os.path.join(os.getcwd(), 'train') 
DATASET_ROOT_VAL = os.path.join(os.getcwd(), 'validation') 
DOWNLOAD_URL_TRAIN = 'https://download.ai-benchmark.com/s/mopDnsMarBnFsdJ/download/denoising_train_jpeg.zip'
DOWNLOAD_URL_VAL = 'https://download.ai-benchmark.com/s/CPcHKibfEcLBj4X/download/denoising_validation_cropped.zip'

def download_file_and_extract(url, extract_path, file_description):
    print(f"Downloading {file_description} from {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            zip_bytes = io.BytesIO(response.read())
        with zipfile.ZipFile(zip_bytes) as zf:
            zf.extractall(path=extract_path)
        print(f"{file_description} download and extraction complete.")
    except Exception as e:
        print(f"Failed to download/extract {file_description}: {e}")
        return False
    return True

def download_data(root_dir_train, root_dir_val):
    train_original_exists = os.path.isdir(os.path.join(root_dir_train, 'original'))
    val_original_exists = os.path.isdir(os.path.join(root_dir_val, 'original'))
    if train_original_exists and val_original_exists:
        print("Data found locally for both train and validation. Skipping download.")
        return
    extract_path = os.getcwd()
    download_file_and_extract(DOWNLOAD_URL_TRAIN, extract_path, "Training Data")
    download_file_and_extract(DOWNLOAD_URL_VAL, extract_path, "Validation Data")

class LemurDataset(Dataset):
    def __init__(self, root_dir, transform=None, crop_size=(512, 512), is_val=False):
        self.noisy_path = os.path.join(root_dir, 'original')
        self.clean_path = os.path.join(root_dir, 'denoised')
        self.transform = transform
        self.crop_size = crop_size
        self.is_val = is_val
        
        self.clean_files = sorted([f for f in os.listdir(self.clean_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.noisy_files = sorted([f for f in os.listdir(self.noisy_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_img = Image.open(os.path.join(self.clean_path, self.clean_files[idx])).convert("RGB")
        noisy_img = Image.open(os.path.join(self.noisy_path, self.noisy_files[idx])).convert("RGB")
        
        w, h = clean_img.size
        target_h, target_w = self.crop_size

        # --- SERVER-SYNC CROP LOGIC ---
        # We crop 2 pixels larger than needed, then center-crop to target.
        # This ensures the model learns on pixels with full 1-pixel neighbors.
        pad_crop_h = target_h + 2
        pad_crop_w = target_w + 2

        if self.is_val:
            i = (h - target_h) // 2
            j = (w - target_w) // 2
            # For validation, direct crop is fine to match previous baselines
            clean_img = F.crop(clean_img, i, j, target_h, target_w)
            noisy_img = F.crop(noisy_img, i, j, target_h, target_w)
        else:
            # Random crop of 258x258
            i = torch.randint(0, h - pad_crop_h + 1, size=(1,)).item()
            j = torch.randint(0, w - pad_crop_w + 1, size=(1,)).item()
            
            clean_img = F.crop(clean_img, i, j, pad_crop_h, pad_crop_w)
            noisy_img = F.crop(noisy_img, i, j, pad_crop_h, pad_crop_w)
            
            # Center crop to 256x256 (Discarding the context rows)
            clean_img = F.center_crop(clean_img, (target_h, target_w))
            noisy_img = F.center_crop(noisy_img, (target_h, target_w))

            # --- Augmentations ---
            if random.random() > 0.5:
                clean_img = F.hflip(clean_img)
                noisy_img = F.hflip(noisy_img)
            if random.random() > 0.5:
                clean_img = F.vflip(clean_img)
                noisy_img = F.vflip(noisy_img)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                clean_img = F.rotate(clean_img, angle)
                noisy_img = F.rotate(noisy_img, angle)

        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
            
        return noisy_img, clean_img

def loader(transform_fn, task=None):
    download_data(DATASET_ROOT_TRAIN, DATASET_ROOT_VAL) 
    final_transform = transform_fn() 
    
    new_crop = (256, 256)
    
    train_set = LemurDataset(DATASET_ROOT_TRAIN, transform=final_transform, crop_size=new_crop, is_val=False)
    test_set = LemurDataset(DATASET_ROOT_VAL, transform=final_transform, crop_size=new_crop, is_val=True)
    
    return (3, new_crop[0], new_crop[1]), 0.0, train_set, test_set
