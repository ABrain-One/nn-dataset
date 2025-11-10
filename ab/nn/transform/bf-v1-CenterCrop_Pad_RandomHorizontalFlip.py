import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.Pad(padding=3, fill=(79, 49, 220), padding_mode='edge'),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
