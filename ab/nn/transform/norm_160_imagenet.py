from torchvision import transforms


def transform(norm):
    """ImageNet-style 160x160 training transform for faster training on large-scale datasets."""
    return transforms.Compose([
        transforms.RandomResizedCrop(160, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*norm),
    ])
