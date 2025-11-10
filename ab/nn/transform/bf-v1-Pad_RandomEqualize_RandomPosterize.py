import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(15, 198, 213), padding_mode='reflect'),
    transforms.RandomEqualize(p=0.16),
    transforms.RandomPosterize(bits=6, p=0.57),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
