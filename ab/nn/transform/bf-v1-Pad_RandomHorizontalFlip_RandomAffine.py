import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(226, 56, 184), padding_mode='constant'),
    transforms.RandomHorizontalFlip(p=0.53),
    transforms.RandomAffine(degrees=7, translate=(0.02, 0.09), scale=(1.1, 1.2), shear=(1.99, 8.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
