import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.13),
    transforms.RandomPosterize(bits=4, p=0.73),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
