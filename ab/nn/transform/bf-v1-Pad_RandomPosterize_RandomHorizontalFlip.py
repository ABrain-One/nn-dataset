import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(219, 105, 67), padding_mode='reflect'),
    transforms.RandomPosterize(bits=4, p=0.38),
    transforms.RandomHorizontalFlip(p=0.68),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
