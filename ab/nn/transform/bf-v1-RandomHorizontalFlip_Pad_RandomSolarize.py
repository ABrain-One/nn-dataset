import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.82),
    transforms.Pad(padding=1, fill=(105, 17, 1), padding_mode='reflect'),
    transforms.RandomSolarize(threshold=240, p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
