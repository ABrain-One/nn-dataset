import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(244, 7, 205), padding_mode='constant'),
    transforms.CenterCrop(size=29),
    transforms.RandomPosterize(bits=5, p=0.25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
