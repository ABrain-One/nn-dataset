import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

def supported_hyperparameters():
    return {'lr','momentum'}


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class CNNEncoder(nn.Module):
    def __init__(self, dims: Tuple[int, int, int]) -> None:
        super().__init__()
        self.dims = dims
        self.seq_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dims[2], dims[2]),
            nn.LayerNorm(dims[2])
        )
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128 * 16),
            nn.LeakyReLU(
# --- auto-closed by AlterCaptionNN ---
))
