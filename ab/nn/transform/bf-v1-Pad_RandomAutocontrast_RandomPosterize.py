import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(70, 201, 240), padding_mode='constant'),
    transforms.RandomAutocontrast(p=0.3),
    transforms.RandomPosterize(bits=7, p=0.6),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
