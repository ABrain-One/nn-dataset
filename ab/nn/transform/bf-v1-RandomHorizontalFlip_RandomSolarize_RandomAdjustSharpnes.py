import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.88),
    transforms.RandomSolarize(threshold=248, p=0.74),
    transforms.RandomAdjustSharpness(sharpness_factor=1.12, p=0.15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
