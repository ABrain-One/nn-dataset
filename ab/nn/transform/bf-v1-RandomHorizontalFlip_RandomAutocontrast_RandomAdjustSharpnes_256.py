import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.84),
    transforms.RandomAutocontrast(p=0.78),
    transforms.RandomAdjustSharpness(sharpness_factor=1.31, p=0.53),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
