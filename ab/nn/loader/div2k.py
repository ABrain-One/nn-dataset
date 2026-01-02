import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import Dataset
from PIL import Image

# Import the standard data directory
from ab.nn.util.Const import data_dir

# 1. Define Constants (Mimicking the MNIST structure)
__norm_mean = (0.4488, 0.4371, 0.4040)
__norm_dev = (1.0, 1.0, 1.0)
__class_quantity = 1
MINIMUM_ACCURACY = 0.0

class DIV2KDataset(Dataset):
    """
    Custom Dataset that works exactly like torchvision.datasets.MNIST
    but handles DIV2K downloading and file loading.
    """
    urls = {
        'train_hr': "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        'valid_hr': "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        'train_lr': "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip",
        'valid_lr': "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip"
    }

    def __init__(self, root, split='train', transform=None, download=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.scale = 2 
        
        # Automatic download logic (Professor's requirement)
        if download:
            self._download()

        # Define paths
        self.hr_dir = os.path.join(self.root, f'DIV2K_{split}_HR')
        self.lr_dir = os.path.join(self.root, f'DIV2K_{split}_LR_bicubic/X{self.scale}')

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.image_filenames = sorted([x for x in os.listdir(self.hr_dir) if x.endswith('.png')])

    def _download(self):
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)
        print(f"Downloading DIV2K {self.split} set...")
        download_and_extract_archive(self.urls[f'{self.split}_hr'], download_root=self.root)
        download_and_extract_archive(self.urls[f'{self.split}_lr'], download_root=self.root)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, f'DIV2K_{self.split}_HR'))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        hr_path = os.path.join(self.hr_dir, img_name)
        lr_path = os.path.join(self.lr_dir, img_name.replace('.png', f'x{self.scale}.png'))

        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')

        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        return lr_img, hr_img

def loader(transform_fn, task):
    """
    Standard Loader Function matching mnist.py signature.
    """
    transform = transform_fn((__norm_mean, __norm_dev))

    # Trigger the auto-download
    train_set = DIV2KDataset(root=data_dir, split='train', transform=transform, download=True)
    test_set = DIV2KDataset(root=data_dir, split='valid', transform=transform, download=True)

    return (__class_quantity,), MINIMUM_ACCURACY, train_set, test_set
