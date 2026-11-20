import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """Stochastic depth per sample."""
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

class ConvBlock(nn.Module):
    """Standard Conv + BN + Activation block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class GatedResidualBlock(nn.Module):
    """
    A block that processes the input through a depthwise separable path,
    applies a sigmoid gate derived from the input (or a separate branch),
    and adds it to the residual stream.
    This combines the efficiency of depthwise convs (Model 1) with a gating
    mechanism that allows the network to modulate feature flow, potentially
    improving gradient flow and feature selection in a single epoch.
    """
    def __init__(self, channels, drop_path_prob=0.0):
        super(GatedResidualBlock, self).__init__()
        self.channels = channels
        
        # Main processing path: Depthwise Separable
        # 1x1 expansion is often better for feature mixing, but let's stick to
        # efficient depthwise + pointwise as in Model 1, but with a gate.
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Gate generation: A 1x1 conv to produce the gate values
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        
        self.act = nn.SiLU(inplace=True)
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        identity = x
        
        # Process
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.pw_conv(out)
        out = self.bn2(out)
        
        # Generate gate from input x (or processed out? Let's use x for a direct gate)
        gate = self.gate_conv(x)
        gate = torch.sigmoid(gate)
        
        # Combine: Identity + (Out * Gate)
        out = identity + (out * gate)
        
        return self.drop_path(out)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        n_classes = out_shape[0] if out_shape else 10
        
        # Derive input channels
        c_in = in_shape[1] if len(in_shape) == 3 and in_shape[0] != 3 and in_shape[-1] != 3 else 3
        if len(in_shape) == 4:
            c_in = in_shape[1]
        # Fallback for CIFAR
        if c_in is None or c_in == 0:
            c_in = 3

        # Architecture Design:
        # We use a channel progression that is distinct from 64-128-256-512.
        # Let's use: 48 -> 96 -> 192 -> 384
        # This is a 2x expansion at each stage.
        
        # Stem: 3 -> 48
        self.stem = ConvBlock(c_in, 48, kernel_size=3, stride=1, padding=1)
        
        # Stage 1: 48 channels, 32x32
        self.stage1 = nn.Sequential(
            GatedResidualBlock(48, drop_path_prob=0.0),
            GatedResidualBlock(48, drop_path_prob=0.0),
            GatedResidualBlock(48, drop_path_prob=0.0),
        )
        
        # Transition 1: 48 -> 96, 32x32 -> 16x16
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 96 channels, 16x16
        self.stage2 = nn.Sequential(
            GatedResidualBlock(96, drop_path_prob=0.0),
            GatedResidualBlock(96, drop_path_prob=0.0),
            GatedResidualBlock(96, drop_path_prob=0.0),
            GatedResidualBlock(96, drop_path_prob=0.0),
        )
        
        # Transition 2: 96 -> 192, 16x16 -> 8x8
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 192 channels, 8x8
        self.stage3 = nn.Sequential(
            GatedResidualBlock(192, drop_path_prob=0.0),
            GatedResidualBlock(192, drop_path_prob=0.0),
            GatedResidualBlock(192, drop_path_prob=0.0),
            GatedResidualBlock(192, drop_path_prob=0.0),
        )
        
        # Transition 3: 192 -> 384, 8x8 -> 4x4
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        # Stage 4: 384 channels, 4x4
        self.stage4 = nn.Sequential(
            GatedResidualBlock(384, drop_path_prob=0.0),
            GatedResidualBlock(384, drop_path_prob=0.0),
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(384, n_classes)
        
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
        
        # Cosine schedule within 1 epoch
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