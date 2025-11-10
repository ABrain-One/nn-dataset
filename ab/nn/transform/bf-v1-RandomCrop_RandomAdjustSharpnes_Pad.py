import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomAdjustSharpness(sharpness_factor=0.57, p=0.11),
    transforms.Pad(padding=0, fill=(67, 98, 102), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
