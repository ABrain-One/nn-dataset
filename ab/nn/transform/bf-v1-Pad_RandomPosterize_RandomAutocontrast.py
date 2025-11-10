import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(217, 226, 101), padding_mode='constant'),
    transforms.RandomPosterize(bits=6, p=0.59),
    transforms.RandomAutocontrast(p=0.47),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
