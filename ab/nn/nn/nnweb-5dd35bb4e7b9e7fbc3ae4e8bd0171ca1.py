import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- MANDATORY FOR EVAL ENGINE ---
def supported_hyperparameters():
    return {'lr', 'momentum'}

# --- Helper Classes ---

class DropPath(nn.Module):
    """Stochastic depth per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return inputs
        keep_prob = 1 - self.drop_prob
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=inputs.dtype, device=inputs.device)
        random_tensor.floor_()
        random_tensor.div_(keep_prob)
        return inputs * random_tensor

class SeparableBlock(nn.Module):
    """
    A block combining depthwise and pointwise convolutions with a residual connection.
    Uses SiLU activation and BatchNorm.
    """
    def __init__(self, channels: int, drop_prob: float = 0.0):
        super().__init__()
        self.channels = channels
        
        # Depthwise Conv
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Pointwise Conv
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.act = nn.SiLU(inplace=True)
        self.drop_path = DropPath(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.pw_conv(out)
        out = self.bn2(out)
        
        out = self.drop_path(out)
        
        return x + out

class TransitionBlock(nn.Module):
    """
    Handles channel expansion and spatial downsampling.
    Uses a 1x1 conv for channel change and AvgPool for spatial reduction.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x

# --- Main Network ---
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device

        # Derive input channels
        # Robustly handle in_shape which might be (C, H, W) or (H, W, C) or (N, C, H, W)
        if len(in_shape) == 4:
            c_in = in_shape[1]
        elif len(in_shape) == 3:
            # Heuristic: if one dim is 3, it's likely channels
            if 3 in in_shape:
                c_in = 3
            else:
                c_in = in_shape[0] # Fallback
        else:
            c_in = 3

        n_classes = out_shape[0] if out_shape else 10

        # Architecture Configuration
        # Avoids standard ResNet/VGG layouts and 64-128-256-512 progression.
        # Uses a lighter, more efficient progression: 48 -> 96 -> 192.
        # This keeps spatial resolution longer (32->16->8) and uses efficient separable convs.
        
        # Stem: 3 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, 32x32 spatial
        # 3 Separable Blocks
        self.stage1 = nn.Sequential(
            SeparableBlock(48, drop_prob=0.0),
            SeparableBlock(48, drop_prob=0.0),
            SeparableBlock(48, drop_prob=0.0),
        )
        
        # Transition 1: 48 -> 96, 32x32 -> 16x16
        self.trans1 = TransitionBlock(48, 96)
        
        # Stage 2: 96 channels, 16x16 spatial
        # 4 Separable Blocks
        self.stage2 = nn.Sequential(
            SeparableBlock(96, drop_prob=0.0),
            SeparableBlock(96, drop_prob=0.0),
            SeparableBlock(96, drop_prob=0.0),
            SeparableBlock(96, drop_prob=0.0),
        )
        
        # Transition 2: 96 -> 192, 16x16 -> 8x8
        self.trans2 = TransitionBlock(96, 192)
        
        # Stage 3: 192 channels, 8x8 spatial
        # 4 Separable Blocks
        self.stage3 = nn.Sequential(
            SeparableBlock(192, drop_prob=0.0),
            SeparableBlock(192, drop_prob=0.0),
            SeparableBlock(192, drop_prob=0.0),
            SeparableBlock(192, drop_prob=0.0),
        )
        
        # Head
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