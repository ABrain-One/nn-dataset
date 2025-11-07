import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.Pad(padding=5, fill=(99, 122, 251), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.39, p=0.65),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
