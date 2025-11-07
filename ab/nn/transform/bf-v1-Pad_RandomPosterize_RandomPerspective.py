import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(180, 212, 190), padding_mode='edge'),
    transforms.RandomPosterize(bits=5, p=0.63),
    transforms.RandomPerspective(distortion_scale=0.19, p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
