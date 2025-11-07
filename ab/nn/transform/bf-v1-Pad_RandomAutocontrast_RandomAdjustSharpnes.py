import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(85, 20, 79), padding_mode='constant'),
    transforms.RandomAutocontrast(p=0.82),
    transforms.RandomAdjustSharpness(sharpness_factor=0.93, p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
