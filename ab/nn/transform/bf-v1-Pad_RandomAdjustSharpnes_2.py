import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(207, 192, 160), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.84, p=0.58),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
