import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class DepthwiseFractalBlock(nn.Module):
    """
    A recursive block that uses depthwise convolutions and channel mixing.
    Depth 1: single DW conv + 1x1 mix
    Depth > 1: recursive structure with parallel branches
    """
    def __init__(self, depth, channels, drop_prob=0.0):
        super().__init__()
        self.depth = depth
        self.channels = channels
        
        # Core: depthwise conv + BN + SiLU + 1x1 channel mix
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        
        self.mix_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn_mix = nn.BatchNorm2d(channels)
        
        if depth > 1:
            self.left = DepthwiseFractalBlock(depth - 1, channels, drop_prob)
            self.right = DepthwiseFractalBlock(depth - 1, channels, drop_prob)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        # Core processing
        y = self.dw_conv(x)
        y = self.bn_dw(y)
        y = self.act(y)
        y = self.mix_conv(y)
        y = self.bn_mix(y)
        
        if self.depth == 1:
            return x + y  # residual
        
        # Recursive: process through left and right branches
        left_out = self.left(x)
        right_out = self.right(x)
        
        # Gated combination
        gate = self.gate(y)
        # gate is (B, C), left_out is (B, C, H, W). 
        # Need to unsqueeze gate to (B, C, 1, 1) for broadcasting
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        combined = gate * left_out + (1 - gate) * right_out
        
        return y + combined

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, depth 2
        self.stage1 = DepthwiseFractalBlock(depth=2, channels=48)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 96 channels, depth 3
        self.stage2 = DepthwiseFractalBlock(depth=3, channels=96)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 192 channels, depth 3
        self.stage3 = DepthwiseFractalBlock(depth=3, channels=192)
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, n_classes)
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
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum-like behavior
        # Model 1 used AdamW with lr=0.002, momentum=0.75
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Store batch size for potential use
        self.batch_size = prm.get('batch', 128)
        
        return self.optimizer
    
    def learn(self, train_data):
        self.train()
        
        # Get total steps
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