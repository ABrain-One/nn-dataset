import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(243, 73, 167), padding_mode='edge'),
    transforms.CenterCrop(size=26),
    transforms.RandomHorizontalFlip(p=0.13),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
