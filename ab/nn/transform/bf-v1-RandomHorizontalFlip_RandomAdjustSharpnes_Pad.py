import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.68),
    transforms.RandomAdjustSharpness(sharpness_factor=1.24, p=0.84),
    transforms.Pad(padding=1, fill=(123, 251, 124), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
