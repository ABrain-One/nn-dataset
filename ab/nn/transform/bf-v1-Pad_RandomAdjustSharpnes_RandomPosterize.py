import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(165, 9, 181), padding_mode='reflect'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.51, p=0.77),
    transforms.RandomPosterize(bits=7, p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
