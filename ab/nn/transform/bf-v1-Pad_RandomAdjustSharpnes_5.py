import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(79, 7, 228), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.94, p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
