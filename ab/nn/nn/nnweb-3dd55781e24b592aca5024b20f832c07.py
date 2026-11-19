import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def supported_hyperparameters():
    return {'lr', 'momentum', 'transform'}

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.mean(dim=(2, 3))  # squeeze
        y = self.fc(y).view(b, c, 1, 1)  # excitation
        return x * y


class GhostModule(nn.Module):
    """Ghost module: generates extra feature maps using cheap linear ops."""
    def __init__(self, in_channels, out_channels, dw_size=3, ratio=2):
        super().__init__()
        self.ghost_channels = out_channels // ratio
        self.ghost_ratio = ratio
        self.ghost_dw_size = dw_size
        self.conv1 = nn.Conv2d(in_channels, self.ghost_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ghost_channels)
        self.conv2 = nn.Conv2d(self.ghost_channels, out_channels, dw_size, padding=dw_size//2, groups=self.ghost_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        return x


class ConvBlock(nn.Module):
    """Standard convolution block with residual connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.act(out)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        # Robustly determine input channels
        # Assume in_shape is (C, H, W) or (H, W, C)
        if len(in_shape) == 3:
            c0, c1, c2 = in_shape
            # Check if (C, H, W)
            if c0 in [1, 3] and c1 == c2:
                c_in = c0
            # Check if (H, W, C)
            elif c2 in [1, 3] and c0 == c1:
                c_in = c2
            # Fallback: assume (C, H, W)
            else:
                c_in = c0
        else:
            c_in = 3 # Default for CIFAR-10 if shape is ambiguous

        h, w = in_shape[1], in_shape[2] if len(in_shape) == 3 else (32, 32)
        num_classes = out_shape[0]
        
        # Entry: 3x3 conv to establish spatial context
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 24 -> 48 channels, spatial size H/2
        self.stage1 = nn.Sequential(
            GhostModule(24, 48, dw_size=3, ratio=2),
            ConvBlock(48, 48, stride=1),
            ConvBlock(48, 48, stride=1),
            nn.MaxPool2d(2)
        )
        
        # Stage 2: 48 -> 96 channels, spatial size H/4
        self.stage2 = nn.Sequential(
            GhostModule(48, 96, dw_size=3, ratio=2),
            ConvBlock(96, 96, stride=1),
            ConvBlock(96, 96, stride=1),
            SEBlock(96, reduction=16),
            nn.MaxPool2d(2)
        )
        
        # Stage 3: 96 -> 192 channels, spatial size H/8
        self.stage3 = nn.Sequential(
            GhostModule(96, 192, dw_size=3, ratio=2),
            ConvBlock(192, 192, stride=1),
            ConvBlock(192, 192, stride=1),
            SEBlock(192, reduction=16),
            nn.MaxPool2d(2)
        )
        
        # Stage 4: 192 -> 288 channels, spatial size H/16 (no pool)
        self.stage4 = nn.Sequential(
            GhostModule(192, 288, dw_size=3, ratio=2),
            ConvBlock(288, 288, stride=1),
            ConvBlock(288, 288, stride=1),
            SEBlock(288, reduction=16)
        )
        
        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(288, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Weight initialization
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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # If input is (N, H, W, C), permute to (N, C, H, W)
        if x.dim() == 4 and x.shape[1] != x.shape[-1] and x.shape[1] not in [1, 3]:
            # Heuristic: if dim 1 is not 1 or 3, and dim -1 is 1 or 3, assume HWCD
            if x.shape[-1] in [1, 3]:
                x = x.permute(0, 3, 1, 2)
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        lr = prm['lr']
        momentum = prm['momentum']
        
        # Use SGD with momentum and Nesterov for stability
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=1e-4,
            nesterov=True
        )
        
        # Cosine annealing within the single epoch
        # Estimate total steps from batch size
        batch_size = prm.get('batch', 128)
        # Assume CIFAR-10 size (50k) for step count estimation
        num_train_samples = 50000
        total_steps = max(1, num_train_samples // batch_size)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps,
            eta_min=lr * 0.01
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.max_steps = total_steps
        return self.optimizer

    def learn(self, train_data):
        self.train()
        step = 0
        for inputs, labels in train_data:
            if step >= self.max_steps:
                break
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            step += 1