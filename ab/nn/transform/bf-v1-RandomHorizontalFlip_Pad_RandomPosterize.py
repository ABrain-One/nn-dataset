import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.86),
    transforms.Pad(padding=1, fill=(169, 100, 172), padding_mode='constant'),
    transforms.RandomPosterize(bits=4, p=0.81),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
