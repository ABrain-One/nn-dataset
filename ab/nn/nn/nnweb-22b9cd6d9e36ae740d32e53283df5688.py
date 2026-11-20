import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) implementation.
    """
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

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid_dim = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FusedBlock(nn.Module):
    """
    A block that fuses a depthwise path and a pointwise path with a residual connection.
    Uses SE for channel re-calibration.
    """
    def __init__(self, channels, expand_ratio=2, drop_prob=0.0):
        super(FusedBlock, self).__init__()
        hidden_dim = int(channels * expand_ratio)
        
        # Depthwise Path: 3x3 DW -> BN -> SiLU
        self.dw_conv = nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(hidden_dim)
        
        # Pointwise Path: 1x1 PW -> BN -> SiLU
        self.pw_conv = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(hidden_dim)
        
        self.act = nn.SiLU(inplace=True)
        
        # Fusion: 1x1 Conv to project back to channels
        self.fusion_conv = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(channels)
        
        # SE Block
        self.se = SEBlock(channels)
        
        # Drop Path
        self.drop_path = DropPath(drop_prob)

    def forward(self, x):
        shortcut = x
        
        # Branch 1: Depthwise
        y1 = self.dw_conv(x)
        y1 = self.bn_dw(y1)
        y1 = self.act(y1)
        
        # Branch 2: Pointwise
        y2 = self.pw_conv(x)
        y2 = self.bn_pw(y2)
        y2 = self.act(y2)
        
        # Combine branches (Sum)
        y = y1 + y2
        
        # Project back
        y = self.fusion_conv(y)
        y = self.bn_fusion(y)
        
        # SE
        y = self.se(y)
        
        # Residual
        y = self.drop_path(y)
        out = x + y
        
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        
        # Determine input channels
        # in_shape is typically (C, H, W) or (H, W, C). CIFAR is 3 channels.
        # We assume standard CIFAR-10 (3, 32, 32) or (32, 32, 3).
        # To be safe, we check for 3.
        if len(in_shape) == 3:
            if in_shape[0] == 3:
                c_in = 3
            elif in_shape[-1] == 3:
                c_in = 3
            else:
                c_in = in_shape[1] # Fallback
        elif len(in_shape) == 4:
            c_in = in_shape[1]
        else:
            c_in = 3
            
        n_classes = out_shape[0] if out_shape else 10
        
        # Architecture Configuration
        # Stem: 3 -> 48
        # Stage 1: 48 ch, 32x32 spatial, 2 blocks
        # Transition: 48 -> 96, Pool 2x
        # Stage 2: 96 ch, 16x16 spatial, 3 blocks
        # Transition: 96 -> 192, Pool 2x
        # Stage 3: 192 ch, 8x8 spatial, 4 blocks
        # Head: GAP -> FC
        
        # This channel progression (48, 96, 192) is distinct from 64-128-256-512.
        
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            FusedBlock(48, expand_ratio=2, drop_prob=0.0),
            FusedBlock(48, expand_ratio=2, drop_prob=0.0)
        )
        
        # Transition 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            FusedBlock(96, expand_ratio=2, drop_prob=0.0),
            FusedBlock(96, expand_ratio=2, drop_prob=0.0),
            FusedBlock(96, expand_ratio=2, drop_prob=0.0)
        )
        
        # Transition 2
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            FusedBlock(192, expand_ratio=2, drop_prob=0.0),
            FusedBlock(192, expand_ratio=2, drop_prob=0.0),
            FusedBlock(192, expand_ratio=2, drop_prob=0.0),
            FusedBlock(192, expand_ratio=2, drop_prob=0.0)
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, n_classes)
        
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