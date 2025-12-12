import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(72, 31, 208), padding_mode='edge'),
    transforms.RandomRotation(degrees=1),
    transforms.RandomAutocontrast(p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
