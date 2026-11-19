import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """Stochastic Depth per sample."""
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

class EfficientBlock(nn.Module):
    """
    Inverted Bottleneck Block with SE gating.
    1x1 Expand -> Depthwise 3x3 -> 1x1 Project -> SE -> Residual
    """
    def __init__(self, channels, expand_ratio=4, drop_prob=0.0, se_ratio=0.25):
        super().__init__()
        hidden_dim = int(channels * expand_ratio)
        
        # Expansion
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # Squeeze and Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(hidden_dim, max(1, int(hidden_dim * se_ratio)), bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(max(1, int(hidden_dim * se_ratio)), hidden_dim, bias=False),
            nn.Sigmoid()
        )
        
        # Projection
        self.conv3 = nn.Conv2d(hidden_dim, channels, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.act = nn.SiLU(inplace=True)
        self.drop_path = DropPath(drop_prob)

    def forward(self, x):
        shortcut = x
        
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        
        # SE Gate
        se = self.se(y)
        # Reshape SE output to match spatial dimensions of y
        se = se.view(y.size(0), y.size(1), 1, 1)
        y = y * se
        
        y = self.conv3(y)
        y = self.bn3(y)
        
        y = self.drop_path(y)
        
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        
        # Derive input channels robustly
        if len(in_shape) == 3:
            if in_shape[0] == 3:
                c_in = 3
            elif in_shape[-1] == 3:
                c_in = 3
            else:
                c_in = in_shape[1]
        elif len(in_shape) == 4:
            c_in = in_shape[1]
        else:
            c_in = 3
            
        n_classes = out_shape[0] if out_shape else 10
        
        # Architecture:
        # Stem: 3 -> 32 (32x32)
        # Stage 1: 32 ch, 32x32, 2 blocks
        # Downsample: 32 -> 64 (16x16)
        # Stage 2: 64 ch, 16x16, 3 blocks
        # Downsample: 64 -> 128 (8x8)
        # Stage 3: 128 ch, 8x8, 3 blocks
        # Downsample: 128 -> 256 (4x4)
        # Stage 4: 256 ch, 4x4, 3 blocks
        # Head: GAP -> FC
        
        # Reduced block counts in stages 3 and 4 from 4 to 3 to save parameters.
        
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 channels
        self.stage1 = nn.Sequential(
            EfficientBlock(32, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(32, expand_ratio=4, drop_prob=0.0)
        )
        
        # Transition 1: 32 -> 64, downsample
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 64 channels
        self.stage2 = nn.Sequential(
            EfficientBlock(64, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(64, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(64, expand_ratio=4, drop_prob=0.0)
        )
        
        # Transition 2: 64 -> 128, downsample
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 128 channels
        self.stage3 = nn.Sequential(
            EfficientBlock(128, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(128, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(128, expand_ratio=4, drop_prob=0.0)
        )
        
        # Transition 3: 128 -> 256, downsample
        self.trans3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 256 channels
        self.stage4 = nn.Sequential(
            EfficientBlock(256, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(256, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(256, expand_ratio=4, drop_prob=0.0)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)
        
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        x = self.trans3(x)
        
        x = self.stage4(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        
        x = self.fc(x)
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