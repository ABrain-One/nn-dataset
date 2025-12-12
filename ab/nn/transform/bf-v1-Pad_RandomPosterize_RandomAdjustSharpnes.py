import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(37, 38, 180), padding_mode='reflect'),
    transforms.RandomPosterize(bits=6, p=0.34),
    transforms.RandomAdjustSharpness(sharpness_factor=1.92, p=0.68),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
