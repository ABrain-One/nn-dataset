import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(195, 235, 223), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.98, p=0.52),
    transforms.RandomHorizontalFlip(p=0.77),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
