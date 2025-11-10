import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.Pad(padding=4, fill=(135, 79, 252), padding_mode='edge'),
    transforms.RandomAutocontrast(p=0.75),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
