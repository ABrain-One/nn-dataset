import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(229, 116, 27), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.37, p=0.29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
