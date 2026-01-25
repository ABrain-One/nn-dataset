"""
BLIP-2 Processor Transform for NN Dataset Framework

This transform uses the BLIP-2 processor from HuggingFace transformers
for image preprocessing suitable for the Blip2Sota model.
"""

from torchvision import transforms
from PIL import Image
import torch

try:
    from transformers import Blip2Processor
    _processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
except Exception:
    _processor = None


def transform(norm):
    """
    Returns a transform function compatible with NN Dataset framework.
    
    For BLIP-2, we need to:
    1. Resize to 224x224
    2. Normalize appropriately
    
    The actual BLIP-2 processor will be used in the model's forward pass.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
