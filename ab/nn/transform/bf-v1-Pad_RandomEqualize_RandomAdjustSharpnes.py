import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(77, 31, 236), padding_mode='symmetric'),
    transforms.RandomEqualize(p=0.53),
    transforms.RandomAdjustSharpness(sharpness_factor=1.73, p=0.65),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
