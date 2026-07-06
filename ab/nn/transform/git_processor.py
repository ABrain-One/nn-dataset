"""
GIT Processor Transform for NN Dataset Framework

Applies minimal preprocessing — only resize and convert to [0,1].
GIT-SOTA's _prep_images() will clamp and pass directly to the
HuggingFace GitProcessor which applies its own internal normalization.
"""

from torchvision import transforms


def transform(norm):
    """
    Returns a transform compatible with the NN Dataset framework.

    Pipeline:
        PIL Image → Resize(224x224) → ToTensor [0,1]

    NO custom Normalize here — the HuggingFace GitProcessor inside
    GIT-SOTA.py handles its own mean/std normalization internally.
    Custom normalization + denormalization would corrupt images to near-zero.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # No Normalize — images stay in [0,1] for _prep_images() clamp
    ])
