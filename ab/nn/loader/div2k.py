import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Net(Dataset):
    """
    Dataset loader for DIV2K Super-Resolution task.
    It pairs Low-Resolution (LR) images with High-Resolution (HR) originals.
    """
    def __init__(self, root=None, train=True, prm=None):
        # We point to the exact location of your unzipped images
        self.root = os.path.expanduser("~/thesis_project/data/DIV2K")
        self.hr_dir = os.path.join(self.root, "DIV2K_valid_HR")
        self.lr_dir = os.path.join(self.root, "DIV2K_valid_LR_bicubic/X4")
        
        # Check if folders exist and get image list
        if os.path.exists(self.hr_dir):
            self.file_names = sorted([f for f in os.listdir(self.hr_dir) if f.endswith('.png')])
        else:
            self.file_names = []
            print(f"Error: Could not find HR directory at {self.hr_dir}")

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # 1. Load the Sharp (HR) image name
        hr_name = self.file_names[idx]
        
        # 2. Open HR image
        hr_img = Image.open(os.path.join(self.hr_dir, hr_name)).convert("RGB")
        
        # 3. Open matching LR image (naming: 0801.png -> 0801x4.png)
        lr_name = hr_name.replace(".png", "x4.png")
        lr_img = Image.open(os.path.join(self.lr_dir, lr_name)).convert("RGB")
        
        # Return both as PyTorch tensors
        return self.transform(lr_img), self.transform(hr_img)
