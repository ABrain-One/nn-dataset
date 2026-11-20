import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEAttention(nn.Module):
    """Lightweight Squeeze-and-Excitation attention"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w.unsqueeze(-1).unsqueeze(-1)

class DilatedFractalBlock(nn.Module):
    """
    Recursive block with dilation.
    Base case: Dilated DW Conv -> BN -> SiLU -> PW Conv -> BN
    Recursive case: Two parallel branches with different dilations,
    combined via learned gating, then core processing with residual.
    """
    def __init__(self, depth, channels, base_dilation=1):
        super().__init__()
        self.depth = depth
        self.channels = channels
        self.base_dilation = base_dilation
        
        # Core processing with dilation
        dil = base_dilation
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=dil, dilation=dil, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        if depth > 1:
            # Two branches with different dilations for multi-scale
            self.branch1 = DilatedFractalBlock(depth - 1, channels, base_dilation)
            self.branch2 = DilatedFractalBlock(depth - 1, channels, base_dilation * 2)
            # Learned gating
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 2),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, x):
        if self.depth > 1:
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            gate = self.gate(x)  # (B, 2)
            gate1 = gate[:, :1].unsqueeze(-1).unsqueeze(-1)
            gate2 = gate[:, 1:].unsqueeze(-1).unsqueeze(-1)
            x = gate1 * out1 + gate2 * out2
        
        # Core with residual
        y = self.dw_conv(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.pw_conv(y)
        y = self.bn2(y)
        
        return x + y

class MultiScaleHead(nn.Module):
    """Head that fuses multi-scale features before classification"""
    def __init__(self, channels, num_classes):
        super().__init__()
        self.se = SEAttention(channels, reduction=4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(channels * 2, num_classes)

    def forward(self, x):
        x = self.se(x)
        avg = self.gap(x).flatten(1)
        mx = self.gmp(x).flatten(1)
        feat = torch.cat([avg, mx], dim=1)
        return self.classifier(feat)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        # Store base learning rate for scheduling
        self.base_lr = prm.get('lr', 0.1)
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 48 with larger kernel for early context
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, depth 2, dilation 1
        self.stage1 = DilatedFractalBlock(depth=2, channels=48, base_dilation=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 96 channels, depth 2, dilation 1
        self.stage2 = DilatedFractalBlock(depth=2, channels=96, base_dilation=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Transition: 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 192 channels, depth 3, dilation 1 (deeper for more capacity)
        self.stage3 = DilatedFractalBlock(depth=3, channels=192, base_dilation=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 192 -> 384
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        # Stage 4: 384 channels, depth 2, dilation 1
        self.stage4 = DilatedFractalBlock(depth=2, channels=384, base_dilation=1)
        
        # Head with multi-scale fusion
        self.head = MultiScaleHead(384, n_classes)
        
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
        # Apply normalization
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.trans3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Update base lr in case it differs (though it should be same as init)
        self.base_lr = prm.get('lr', 0.1)
        
        # SGD with momentum - more stable for 1-epoch with cosine schedule
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.base_lr,
            momentum=prm['momentum'],
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
        
        # Cosine schedule: warmup for first 5% then cosine decay
        warmup_steps = max(1, int(total_steps * 0.05))
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Learning rate schedule
            if i < warmup_steps:
                lr_scale = (i + 1) / warmup_steps
            else:
                progress = (i - warmup_steps) / max(1, total_steps - warmup_steps)
                lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * lr_scale
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()