import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.Pad(padding=3, fill=(76, 246, 30), padding_mode='symmetric'),
    transforms.RandomPosterize(bits=8, p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
