import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1, stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = False, act: Optional[nn.Module] = nn.GELU()):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class EfficientViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, act: Optional[nn.Module] = nn.GELU()):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act,
            nn.Linear(int(dim * mlp_ratio), dim),
            act,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_norm2 = self.norm2(x_flat)
        x_flat = x_flat + self.mlp(x_norm2)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, depth: int = 6, mlp_ratio: float = 4.0, act: Optional[nn.Module] = nn.GELU()):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=depth, padding=depth // 2, groups=dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.pw = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = act
        self.pw2 = nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.depthwise(x)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)
        x_mlp = self.pw2(self.act(self.pw(x_norm)))
        x_flat = x_flat + x_mlp
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.num_classes = out_shape[0]
        dropout: float = prm['dropout']
        
        # Determine input size from in_shape (H, W, C) or (C, H, W)
        # CIFAR is 32x32x3. in_shape is usually (C, H, W) in PyTorch contexts but let's be safe.
        if len(in_shape) == 3:
            if in_shape[0] <= 10: # Likely C, H, W
                in_ch, H, W = in_shape
            else: # Likely H, W, C
                H, W, in_ch = in_shape
        else:
            in_ch = 3
            H = W = 32

        # Architecture: Hybrid CNN-Transformer (EfficientViT / ConvNeXt style)
        # Stage 1: Stem
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, 64, kernel_size=7, stride=2),
            ConvBNAct(64, 64, kernel_size=3, stride=1),
        )
        
        # Stage 2: Downsample + ConvNeXt Blocks
        self.stage2 = nn.Sequential(
            ConvBNAct(64, 128, kernel_size=3, stride=2),
            ConvNeXtBlock(128, depth=7),
            ConvNeXtBlock(128, depth=7),
        )
        
        # Stage 3: Downsample + Transformer Blocks
        self.stage3 = nn.Sequential(
            ConvBNAct(128, 256, kernel_size=3, stride=2),
            EfficientViTBlock(256, num_heads=4),
            EfficientViTBlock(256, num_heads=4),
        )
        
        # Stage 4: Downsample + Transformer Blocks
        self.stage4 = nn.Sequential(
            ConvBNAct(256, 512, kernel_size=3, stride=2),
            EfficientViTBlock(512, num_heads=8),
            EfficientViTBlock(512, num_heads=8),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, self.num_classes)
        )
        
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}