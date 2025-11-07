import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.61),
    transforms.RandomAdjustSharpness(sharpness_factor=1.47, p=0.45),
    transforms.RandomPosterize(bits=5, p=0.77),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
