import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.87),
    transforms.RandomAdjustSharpness(sharpness_factor=1.99, p=0.73),
    transforms.RandomPerspective(distortion_scale=0.19, p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
