import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(91, 244, 152), padding_mode='symmetric'),
    transforms.RandomRotation(degrees=16),
    transforms.RandomHorizontalFlip(p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
