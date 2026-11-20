import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_bn=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class GatedResidualBlock(nn.Module):
    """
    A block that uses a gating mechanism to control the residual stream.
    It processes the input through a depthwise separable conv path and a 1x1 path,
    combines them, and then applies a sigmoid gate to the residual connection.
    """
    def __init__(self, channels, drop_path_prob=0.0):
        super(GatedResidualBlock, self).__init__()
        self.channels = channels
        
        # Main processing path: Depthwise Separable
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        
        # Gating path
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gate_act = nn.Sigmoid()
        
        self.drop_path = nn.Identity()
        if drop_path_prob > 0.0:
            self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        identity = x
        
        # Process
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.pw_conv(out)
        out = self.bn2(out)
        
        # Gate
        gate = self.gate_conv(x)
        gate = self.gate_act(gate)
        
        # Combine: Identity + (Out * Gate)
        out = identity + (out * gate)
        
        return self.drop_path(out)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        random_tensor.div_(keep_prob)
        return x * random_tensor

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        n_classes = out_shape[0] if out_shape else 10
        
        # Input size is expected to be 32x32 for CIFAR, but we derive from in_shape if possible
        # in_shape is usually (C, H, W) or (H, W, C). Let's assume standard CIFAR (3, 32, 32)
        # We will adapt the number of stages based on input size if it's not 32, 
        # but for CIFAR-10, 32 is standard.
        
        # Architecture:
        # Stem: 3 -> 48
        # Stage 1: 48 -> 96 (2x downsample)
        # Stage 2: 96 -> 192 (2x downsample)
        # Stage 3: 192 -> 384 (2x downsample)
        # Head: GAP -> FC
        
        # We use a unique channel progression: 48, 96, 192, 384
        # This avoids the standard 64-128-256-512.
        
        self.stem = ConvBlock(3, 48, kernel_size=3, stride=1, padding=1)
        
        # Stage 1: 48 -> 96
        self.stage1 = nn.Sequential(
            GatedResidualBlock(48),
            GatedResidualBlock(48),
            ConvBlock(48, 96, kernel_size=3, stride=2, padding=1),
            GatedResidualBlock(96),
            GatedResidualBlock(96),
        )
        
        # Stage 2: 96 -> 192
        self.stage2 = nn.Sequential(
            GatedResidualBlock(96),
            GatedResidualBlock(96),
            ConvBlock(96, 192, kernel_size=3, stride=2, padding=1),
            GatedResidualBlock(192),
            GatedResidualBlock(192),
        )
        
        # Stage 3: 192 -> 384
        self.stage3 = nn.Sequential(
            GatedResidualBlock(192),
            GatedResidualBlock(192),
            ConvBlock(192, 384, kernel_size=3, stride=2, padding=1),
            GatedResidualBlock(384),
            GatedResidualBlock(384),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(384, n_classes)
        
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Store batch size
        self.batch_size = prm.get('batch', 128)
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
        
        # Cosine schedule within 1 epoch
        # Start at lr, end at 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=0.0
        )
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            
            self.optimizer.step()
            scheduler.step()