import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.88),
    transforms.RandomPosterize(bits=4, p=0.42),
    transforms.Pad(padding=3, fill=(52, 163, 140), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
