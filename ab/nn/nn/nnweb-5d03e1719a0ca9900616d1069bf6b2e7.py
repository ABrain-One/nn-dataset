import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DoubleConvBlock(nn.Module):
    """Standard double convolution block with residual connection."""
    def __init__(self, in_ch, out_ch):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        
        # Identity mapping for residual connection
        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out)
        
        return out + identity

class ChannelMixingBlock(nn.Module):
    """A block that mixes channels via 1x1 convolutions and spatial mixing via 3x3 depthwise convolutions."""
    def __init__(self, channels, expansion=4):
        super(ChannelMixingBlock, self).__init__()
        mid_ch = channels * expansion
        
        # 1x1 expansion
        self.conv1 = nn.Conv2d(channels, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        
        # 3x3 depthwise for spatial mixing
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        
        # 1x1 projection
        self.conv3 = nn.Conv2d(mid_ch, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.act = nn.SiLU(inplace=True)
        
        # Residual shortcut
        self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        return out + identity

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        # in_shape is usually (C, H, W)
        # Fix: Robustly determine input channels
        if len(in_shape) == 3:
            c_in = in_shape[0]
        else:
            c_in = 3  # Default to RGB if shape is ambiguous or 4D (B, C, H, W)

        # Handle out_shape which might be (num_classes,) or just an int
        if isinstance(out_shape, int):
            n_classes = out_shape
        else:
            n_classes = out_shape[0] if len(out_shape) > 0 else 10

        # Architecture Design:
        # Avoid 64-128-256-512 progression.
        # Use a progression like 48 -> 96 -> 192 -> 384.
        
        # Stem: 3x3 conv, no pooling (to preserve resolution for small images like CIFAR)
        stem_ch = 48
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, stem_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, 2 DoubleConvBlocks
        self.stage1 = nn.Sequential(
            DoubleConvBlock(stem_ch, stem_ch),
            DoubleConvBlock(stem_ch, stem_ch)
        )
        
        # Transition 1: Pooling + Channel change to 96
        self.trans1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(stem_ch, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 96 channels, 2 ChannelMixingBlocks
        self.stage2 = nn.Sequential(
            ChannelMixingBlock(96),
            ChannelMixingBlock(96)
        )
        
        # Transition 2: Pooling + Channel change to 192
        self.trans2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 192 channels, 2 DoubleConvBlocks
        self.stage3 = nn.Sequential(
            DoubleConvBlock(192, 192),
            DoubleConvBlock(192, 192)
        )
        
        # Transition 3: Pooling + Channel change to 384
        self.trans3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        # Stage 4: 384 channels, 1 ChannelMixingBlock
        self.stage4 = nn.Sequential(
            ChannelMixingBlock(384)
        )
        
        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, n_classes)
        )
        
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.trans3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        self.batch_size = prm.get('batch', 128)
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
        
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
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            
            self.optimizer.step()
            scheduler.step()