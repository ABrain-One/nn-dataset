import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.51),
    transforms.RandomAutocontrast(p=0.8),
    transforms.Pad(padding=1, fill=(42, 133, 100), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
