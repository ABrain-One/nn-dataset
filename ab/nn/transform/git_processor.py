"""
GIT Processor Transform for NN Dataset Framework

This transform uses appropriate preprocessing for Microsoft's GIT model
for image captioning.
"""

from torchvision import transforms

try:
    from transformers import GitProcessor
    _processor = GitProcessor.from_pretrained("microsoft/git-large-coco")
except Exception:
    _processor = None


def transform(norm):
    """
    Returns a transform function compatible with NN Dataset framework.
    
    For GIT, we need to:
    1. Resize to 224x224
    2. Normalize appropriately
    
    The actual GIT processor will be used in the model's forward pass.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
