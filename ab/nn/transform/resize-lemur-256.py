import torchvision.transforms as T
import torch

def transform(norm: tuple = None) -> T.Compose:
    """
    Defines the standard 256x256 resize and tensor conversion for denoising task.
    """
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
