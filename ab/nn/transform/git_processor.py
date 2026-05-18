"""
Lightweight GIT Transform for NN Dataset Framework
"""

from torchvision import transforms

def transform(norm):
    """
    Super lightweight transform to avoid OOM in DataLoader workers.
    Standard GIT preprocessing: Resize to 224x224 and basic normalization.
    """
    # Microsoft GIT (large) expects 224x224 and standard ImageNet-like normalization
    # but we'll do the normalization on the GPU for speed and RAM efficiency.
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
