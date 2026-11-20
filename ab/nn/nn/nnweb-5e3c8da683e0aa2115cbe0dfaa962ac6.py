import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class GatedFractalBlock(nn.Module):
    """
    A recursive depthwise convolution block.
    Base case: DW Conv -> BN -> SiLU -> PW Conv -> BN -> Residual
    Recursive case: Two sub-blocks, fused via a learned gate (SE-like), then core.
    """
    def __init__(self, depth, channels):
        super().__init__()
        self.depth = depth
        self.channels = channels
        
        # Core processing (always present)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        if depth > 1:
            self.left = GatedFractalBlock(depth - 1, channels)
            self.right = GatedFractalBlock(depth - 1, channels)
            # Learned gate for fusing left and right branches
            # Global average pool -> FC -> Sigmoid
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels),
                nn.Sigmoid()
            )

    def forward(self, x):
        if self.depth > 1:
            left_out = self.left(x)
            right_out = self.right(x)
            # Compute gate based on input x
            gate = self.gate(x)
            gate = gate.unsqueeze(-1).unsqueeze(-1) # Expand for broadcasting
            # Fuse branches
            x = gate * left_out + (1 - gate) * right_out
        
        # Core
        y = self.dw_conv(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.pw_conv(y)
        y = self.bn2(y)
        
        # Residual connection
        return x + y

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block to enhance channel features.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = torch.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 64
        # Use a slightly larger kernel to capture context early
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 64 channels, Depth 2 Fractal
        self.stage1 = GatedFractalBlock(depth=2, channels=64)
        self.se1 = SEBlock(64)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 64 -> 128
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 128 channels, Depth 3 Fractal
        self.stage2 = GatedFractalBlock(depth=3, channels=128)
        self.se2 = SEBlock(128)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 128 -> 256
        self.trans2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 256 channels, Depth 3 Fractal
        self.stage3 = GatedFractalBlock(depth=3, channels=256)
        self.se3 = SEBlock(256)
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, n_classes)
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
        x = self.se1(x)
        x = self.pool1(x)
        
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.se2(x)
        x = self.pool2(x)
        
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.se3(x)
        
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW is the key to Model 1's success
        # Map momentum to beta1
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Store batch size for potential use, though not strictly needed for AdamW
        self.batch_size = prm.get('batch', 128)
        
        return self.optimizer
    
    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
        
        # Cosine Annealing Schedule
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            
            self.optimizer.step()
            scheduler.step()