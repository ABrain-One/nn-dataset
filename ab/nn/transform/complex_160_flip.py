import numpy as np
import torch
import torchvision.transforms as transforms


class NormalizeToFloat:
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32) / 255.0
        elif isinstance(x, torch.Tensor):
            x = x.float() / 255.0
        else:
            x = np.array(x).astype(np.float32) / 255.0
        return x


class ToComplex64:
    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x.type(torch.complex64)


def transform(norm):
    return transforms.Compose([
        transforms.RandomResizedCrop(160),
        transforms.RandomHorizontalFlip(),
        NormalizeToFloat(),
        transforms.ToTensor(),
        ToComplex64(),
    ])
