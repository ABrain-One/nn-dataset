import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(23, 192, 89), padding_mode='reflect'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.53, p=0.35),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
