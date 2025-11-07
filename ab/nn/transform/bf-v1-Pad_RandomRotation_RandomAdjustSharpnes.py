import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(60, 137, 249), padding_mode='edge'),
    transforms.RandomRotation(degrees=8),
    transforms.RandomAdjustSharpness(sharpness_factor=1.23, p=0.62),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
