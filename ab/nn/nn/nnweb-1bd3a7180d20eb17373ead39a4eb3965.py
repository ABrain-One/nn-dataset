import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) implementation."""
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

class ConvBnAct(nn.Module):
    """Standard Convolution + BatchNorm + Activation block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, act=nn.SiLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class InvertedResidual(nn.Module):
    """
    MobileNetV2-style Inverted Residual Block.
    Expansion (1x1) -> Depthwise (3x3) -> Projection (1x1).
    """
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4, drop_path_prob=0.0):
        super(InvertedResidual, self).__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden_dim = int(round(in_ch * expand_ratio))
        
        self.conv1 = ConvBnAct(in_ch, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBnAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim)
        self.conv3 = ConvBnAct(hidden_dim, out_ch, kernel_size=1, stride=1, padding=0, act=nn.Identity)
        
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        if self.use_residual:
            x = self.drop_path(x) + input_x
        else:
            x = self.drop_path(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        n_classes = out_shape[0] if out_shape else 10
        
        # Derive spatial dimensions if needed, though CIFAR is 32x32.
        # in_shape is typically (C, H, W) or (H, W, C).
        # We assume standard CIFAR input 3x32x32 for the architecture design,
        # but the code is generic enough to handle the tensor flow.
        
        # Architecture:
        # Stem: 3 -> 24
        # Stage 1: 24 -> 48 (stride 1)
        # Stage 2: 48 -> 96 (stride 2)
        # Stage 3: 96 -> 192 (stride 2)
        # Stage 4: 192 -> 384 (stride 2)
        # Head: GAP -> FC
        
        # Channel progression: 24, 48, 96, 192, 384
        # This is distinct from 64-128-256-512 and the previous 48-96-192-384 design
        # by using Inverted Residuals (MobileNetV2 style) instead of Gated Residuals.
        
        self.stem = ConvBnAct(3, 24, kernel_size=3, stride=1, padding=1)
        
        # Stage 1: 24 -> 48, stride 1
        self.stage1 = nn.Sequential(
            InvertedResidual(24, 24, stride=1, expand_ratio=4, drop_path_prob=0.1),
            InvertedResidual(24, 48, stride=1, expand_ratio=4, drop_path_prob=0.1),
        )
        
        # Stage 2: 48 -> 96, stride 2
        self.stage2 = nn.Sequential(
            InvertedResidual(48, 48, stride=2, expand_ratio=4, drop_path_prob=0.2),
            InvertedResidual(48, 48, stride=1, expand_ratio=4, drop_path_prob=0.2),
            InvertedResidual(48, 96, stride=1, expand_ratio=4, drop_path_prob=0.2),
        )
        
        # Stage 3: 96 -> 192, stride 2
        self.stage3 = nn.Sequential(
            InvertedResidual(96, 96, stride=2, expand_ratio=4, drop_path_prob=0.3),
            InvertedResidual(96, 96, stride=1, expand_ratio=4, drop_path_prob=0.3),
            InvertedResidual(96, 192, stride=1, expand_ratio=4, drop_path_prob=0.3),
        )
        
        # Stage 4: 192 -> 384, stride 2
        self.stage4 = nn.Sequential(
            InvertedResidual(192, 192, stride=2, expand_ratio=4, drop_path_prob=0.4),
            InvertedResidual(192, 192, stride=1, expand_ratio=4, drop_path_prob=0.4),
            InvertedResidual(192, 384, stride=1, expand_ratio=4, drop_path_prob=0.4),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(384, n_classes)
        
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
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