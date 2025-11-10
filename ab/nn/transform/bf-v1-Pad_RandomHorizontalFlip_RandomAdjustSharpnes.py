import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(198, 7, 128), padding_mode='constant'),
    transforms.RandomHorizontalFlip(p=0.36),
    transforms.RandomAdjustSharpness(sharpness_factor=1.89, p=0.42),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
