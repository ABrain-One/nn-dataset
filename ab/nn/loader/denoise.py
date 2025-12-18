import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import urllib.request 
import zipfile 
import io
import logging

DATASET_ROOT_TRAIN = os.path.join(os.getcwd(), 'train') 
DATASET_ROOT_VAL = os.path.join(os.getcwd(), 'validation') 
DOWNLOAD_URL_TRAIN = 'https://download.ai-benchmark.com/s/mopDnsMarBnFsdJ/download/denoising_train_jpeg.zip'
DOWNLOAD_URL_VAL = 'https://download.ai-benchmark.com/s/CPcHKibfEcLBj4X/download/denoising_validation_cropped.zip'

# --- DOWNLOAD HELPER ---
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
    """Checks if data exists, and if not, downloads and extracts both train and val sets."""
    
    train_original_exists = os.path.isdir(os.path.join(root_dir_train, 'original'))
    val_original_exists = os.path.isdir(os.path.join(root_dir_val, 'original'))
    
    if train_original_exists and val_original_exists:
        print("Data found locally for both train and validation. Skipping download.")
        return
        
    print("Data not found or incomplete. Starting full dataset download...")

    extract_path = os.getcwd()
    os.makedirs(extract_path, exist_ok=True)
    
    success_train = download_file_and_extract(DOWNLOAD_URL_TRAIN, extract_path, "Training Data")
    if not success_train: return
    
    success_val = download_file_and_extract(DOWNLOAD_URL_VAL, extract_path, "Validation Data")
    if not success_val: return
    
    print("\nAll data files are now present in the correct directory.")

class LemurDataset(Dataset):
    """
    Custom Dataset class that maps 'denoised' (input) to 'original' (target).
    """
    def __init__(self, root_dir, transform=None):
        
        self.noisy_path = os.path.join(root_dir, 'denoised')
        self.clean_path = os.path.join(root_dir, 'original')
        
        if os.path.exists(self.clean_path):
            self.clean_files = sorted([f for f in os.listdir(self.clean_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            self.noisy_files = sorted([f for f in os.listdir(self.noisy_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            self.clean_files = []
            self.noisy_files = []
            
        self.transform = transform

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        try:
            clean_p = os.path.join(self.clean_path, self.clean_files[idx])
            noisy_p = os.path.join(self.noisy_path, self.noisy_files[idx])
            
            clean_img = Image.open(clean_p).convert("RGB")
            noisy_img = Image.open(noisy_p).convert("RGB")
            
            if self.transform:
                clean_img = self.transform(clean_img)
                noisy_img = self.transform(noisy_img)
            
            return noisy_img, clean_img
            
        except Exception as e:
            # --- FIX IMPLEMENTED HERE: Added logging ---
            
            # Log the full error, including the specific files that failed
            logging.warning(
                f"Data loading failed at index {idx}. Files: {clean_p}, {noisy_p}. "
                f"Error: {e}. Returning zero tensors."
            )
            
            # Fallback for corrupted files (unchanged)
            return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256)

def loader(transform_fn, task=None):
    """
    CRITICAL: Instantiates the train and test sets using their distinct folder paths.
    """
    
    download_data(DATASET_ROOT_TRAIN, DATASET_ROOT_VAL) 
    
    final_transform = transform_fn() 

    train_set = LemurDataset(DATASET_ROOT_TRAIN, transform=final_transform)
    test_set = LemurDataset(DATASET_ROOT_VAL, transform=final_transform)

    return (3, 256, 256), 0.0, train_set, test_set
