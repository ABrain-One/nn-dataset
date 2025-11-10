import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.15),
    transforms.Pad(padding=3, fill=(74, 222, 154), padding_mode='constant'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.93, p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
