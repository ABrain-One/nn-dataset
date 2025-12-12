import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomAdjustSharpness(sharpness_factor=0.84, p=0.18),
    transforms.RandomPosterize(bits=7, p=0.58),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
