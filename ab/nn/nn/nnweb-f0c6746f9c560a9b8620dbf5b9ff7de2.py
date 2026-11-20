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

class MBConvBlock(nn.Module):
    """
    MobileNetV2 style block with an added 1x1 pointwise convolution at the end.
    This creates a 'Double-Pointwise' structure (Expansion -> Depthwise -> Projection -> Projection).
    The final 1x1 allows for channel mixing before the residual addition, which can help with feature decorrelation.
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, drop_prob=0.0):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        
        hidden_dim = int(in_channels * expand_ratio)
        
        # Expansion
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_dim)
        
        # Projection (1x1)
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Extra 1x1 for mixing (The "Double-Pointwise" twist)
        self.mix_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.mix_bn = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True)
        self.drop_path = DropPath(drop_prob)

    def forward(self, x):
        input_x = x
        
        # Expand
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.act(out)
        
        # Depthwise
        out = self.dw_conv(out)
        out = self.dw_bn(out)
        out = self.act(out)
        
        # Project
        out = self.project_conv(out)
        out = self.project_bn(out)
        
        # Mix (Extra 1x1)
        out = self.mix_conv(out)
        out = self.mix_bn(out)
        
        out = self.drop_path(out)
        
        if self.use_residual:
            out = out + input_x
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        
        # Robust channel detection
        c_in = 3
        if len(in_shape) == 3:
            if in_shape[0] == 3: c_in = 3
            elif in_shape[-1] == 3: c_in = 3
            else: c_in = in_shape[1]
        elif len(in_shape) == 4:
            c_in = in_shape[1]
            
        n_classes = out_shape[0] if out_shape else 10

        # Architecture Design:
        # Goal: Beat 77.14% (Design 1) and 74.34% (Model to Improve).
        # Strategy: 
        # 1. Use a slightly deeper network with more capacity but efficient blocks.
        # 2. Use a unique channel progression: 40 -> 80 -> 160 -> 320. (Avoids 64-128-256-512).
        # 3. Use MBConv blocks with an extra 1x1 "mixing" layer to increase representational power without massive parameter cost.
        # 4. Use SiLU activation (from Model 1/Design 1).
        # 5. Use BatchNorm.
        # 6. Use AvgPool for downsampling (from Model 1).
        
        # Stem: 3 -> 40
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 40 channels, 32x32 -> 16x16
        # 2 blocks
        self.stage1 = nn.Sequential(
            MBConvBlock(40, 40, stride=1, expand_ratio=4, drop_prob=0.0),
            MBConvBlock(40, 80, stride=2, expand_ratio=4, drop_prob=0.0)
        )
        
        # Stage 2: 80 channels, 16x16 -> 8x8
        # 3 blocks
        self.stage2 = nn.Sequential(
            MBConvBlock(80, 80, stride=1, expand_ratio=4, drop_prob=0.0),
            MBConvBlock(80, 80, stride=1, expand_ratio=4, drop_prob=0.0),
            MBConvBlock(80, 160, stride=2, expand_ratio=4, drop_prob=0.0)
        )
        
        # Stage 3: 160 channels, 8x8 -> 4x4
        # 4 blocks
        self.stage3 = nn.Sequential(
            MBConvBlock(160, 160, stride=1, expand_ratio=4, drop_prob=0.0),
            MBConvBlock(160, 160, stride=1, expand_ratio=4, drop_prob=0.0),
            MBConvBlock(160, 160, stride=1, expand_ratio=4, drop_prob=0.0),
            MBConvBlock(160, 320, stride=2, expand_ratio=4, drop_prob=0.0)
        )
        
        # Head
        # GAP
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        # 320 features
        self.fc = nn.Linear(320, n_classes)
        
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
        x = self.stage2(x)
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