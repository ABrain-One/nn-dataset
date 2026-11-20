import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

# --- MANDATORY FOR EVAL ENGINE ---
def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

# --- Building Blocks ---

class MixBlock(nn.Module):
    """
    A single MixNet-style block:
    1. Expand (1x1)
    2. Branch A: Depthwise 3x3 + Project (1x1)
    3. Branch B: Depthwise 5x5 + Project (1x1)
    4. Concat + Project (1x1)
    Residual connection if dimensions match.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expand
        hidden_dim = in_channels * 4
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        # Branch A (Small Kernel)
        self.dw_a = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=stride, groups=hidden_dim, bias=False)
        self.bn_a = nn.BatchNorm2d(hidden_dim)
        self.proj_a = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn_proj_a = nn.BatchNorm2d(out_channels)
        
        # Branch B (Large Kernel)
        # If stride > 1, 5x5 with stride 2 is expensive but manageable for small spatial dims.
        # If stride == 1, 5x5 is fine.
        self.dw_b = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, stride=stride, groups=hidden_dim, bias=False)
        self.bn_b = nn.BatchNorm2d(hidden_dim)
        self.proj_b = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn_proj_b = nn.BatchNorm2d(out_channels)
        
        # Combine
        self.combine_proj = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn_combine = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.expand(x)
        
        # Branch A
        a = self.dw_a(x)
        a = self.bn_a(a)
        a = self.act(a)
        a = self.proj_a(a)
        a = self.bn_proj_a(a)
        
        # Branch B
        b = self.dw_b(x)
        b = self.bn_b(b)
        b = self.act(b)
        b = self.proj_b(b)
        b = self.bn_proj_b(b)
        
        # Concat and Project
        x = torch.cat([a, b], dim=1)
        x = self.combine_proj(x)
        x = self.bn_combine(x)
        
        if self.use_residual:
            x = x + input
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_channels = in_shape[1]
        num_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 64, 32x32 -> 16x16
        # Using stride 2 in stem to reduce spatial resolution early
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 16x16, 64 channels
        # 2 blocks, no stride change (stride=1)
        self.stage1 = nn.Sequential(
            MixBlock(64, 64, stride=1),
            MixBlock(64, 64, stride=1)
        )
        
        # Transition: 64 -> 128, 16x16 -> 8x8
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 8x8, 128 channels
        # 3 blocks, first has stride 1 (input is already downsampled by trans), others stride 1
        # Wait, if we downsample in transition, the blocks inside stage should be stride 1.
        # Let's do: Trans (stride 2) -> Stage (stride 1 blocks)
        self.stage2 = nn.Sequential(
            MixBlock(128, 128, stride=1),
            MixBlock(128, 128, stride=1),
            MixBlock(128, 128, stride=1)
        )
        
        # Transition: 128 -> 256, 8x8 -> 4x4
        self.trans2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 4x4, 256 channels
        # 3 blocks
        self.stage3 = nn.Sequential(
            MixBlock(256, 256, stride=1),
            MixBlock(256, 256, stride=1),
            MixBlock(256, 256, stride=1)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply normalization if needed
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with momentum for stability in 1 epoch
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
        
        # Store batch size
        self.batch_size = prm.get('batch', 128)
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        total_steps = len(train_data)
        
        # Cosine Annealing within 1 epoch
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