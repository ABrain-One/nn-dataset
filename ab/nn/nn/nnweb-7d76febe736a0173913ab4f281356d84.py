import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
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

class MultiScaleDWBlock(nn.Module):
    """
    A block that processes input through parallel depthwise convolutions of different sizes (3x3, 5x5)
    and a 1x1 projection, combining them via a learned linear combination.
    This captures multi-scale local features efficiently.
    """
    def __init__(self, channels, expand_ratio=4, drop_prob=0.0):
        super().__init__()
        self.channels = channels
        hidden_dim = int(channels * expand_ratio)

        # Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        # Parallel Multi-Scale Depthwise Convs
        # We use 3x3 and 5x5 depthwise convs.
        self.dw_3x3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.dw_5x5 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim, bias=False)
        
        # Shared BN and Act for the parallel branches? Or separate?
        # Let's use separate BNs to allow different statistics, then sum.
        self.bn_3x3 = nn.BatchNorm2d(hidden_dim)
        self.bn_5x5 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.SiLU(inplace=True)

        # Projection back
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.drop_path = DropPath(drop_prob)

    def forward(self, x):
        identity = x
        
        y = self.expand(x)
        
        # Parallel branches
        y1 = self.dw_3x3(y)
        y1 = self.bn_3x3(y1)
        
        y2 = self.dw_5x5(y)
        y2 = self.bn_5x5(y2)
        
        # Combine branches (Sum + Act)
        y = y1 + y2
        y = self.act(y)
        
        y = self.project(y)
        y = self.drop_path(y)
        
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        
        # Robustly derive input channels
        c_in = 3
        if len(in_shape) == 3:
            # Assume (C, H, W) or (H, W, C)
            if in_shape[0] == 3:
                c_in = 3
            elif in_shape[-1] == 3:
                c_in = 3
            else:
                c_in = in_shape[1] # Fallback
        elif len(in_shape) == 4:
            c_in = in_shape[1]
            
        n_classes = out_shape[0] if out_shape else 10

        # Architecture:
        # Stem: 3 -> 48
        # Stage 1: 48 channels, 32x32. 2 MultiScaleDWBlocks.
        # Transition: 48 -> 96, 32->16 (AvgPool)
        # Stage 2: 96 channels, 16x16. 3 MultiScaleDWBlocks.
        # Transition: 96 -> 192, 16->8 (AvgPool)
        # Stage 3: 192 channels, 8x8. 4 MultiScaleDWBlocks.
        # Head: GAP -> Linear
        
        # This channel progression (48, 96, 192) avoids the 64-128-256-512 pattern.
        # The block structure (Multi-scale DW) is distinct from Fractal (recursive) and standard EfficientNet (single DW).
        
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            MultiScaleDWBlock(48, expand_ratio=4, drop_prob=0.0),
            MultiScaleDWBlock(48, expand_ratio=4, drop_prob=0.0)
        )
        
        # Transition 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            MultiScaleDWBlock(96, expand_ratio=4, drop_prob=0.0),
            MultiScaleDWBlock(96, expand_ratio=4, drop_prob=0.0),
            MultiScaleDWBlock(96, expand_ratio=4, drop_prob=0.0)
        )
        
        # Transition 2
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            MultiScaleDWBlock(192, expand_ratio=4, drop_prob=0.0),
            MultiScaleDWBlock(192, expand_ratio=4, drop_prob=0.0),
            MultiScaleDWBlock(192, expand_ratio=4, drop_prob=0.0),
            MultiScaleDWBlock(192, expand_ratio=4, drop_prob=0.0)
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, n_classes)
        
        self._init_weights()
        self.to(device)

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
        x = self.stem(x)
        
        x = self.stage1(x)
        x = self.trans1(x)
        
        x = self.stage2(x)
        x = self.trans2(x)
        
        x = self.stage3(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Using the same optimizer setup as the model to improve (which is fixed)
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