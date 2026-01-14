import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.Pad(padding=1, fill=(46, 101, 141), padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(p=0.64),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
