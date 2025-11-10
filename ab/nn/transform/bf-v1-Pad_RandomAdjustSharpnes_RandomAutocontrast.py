import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(54, 212, 70), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.49, p=0.22),
    transforms.RandomAutocontrast(p=0.26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
