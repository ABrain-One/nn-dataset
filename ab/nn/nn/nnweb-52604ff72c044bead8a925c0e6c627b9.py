import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    """Squeeze-and-Excite block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FractalDWBlock(nn.Module):
    """
    Recursive Depthwise Convolution Block.
    Base case: DW Conv -> BN -> SiLU -> 1x1 Conv (with SE) -> BN
    Recursive case: Parallel branches (Left, Right) averaged, then passed through core.
    This structure mirrors Model 1's success but adds SE attention to the pointwise projection.
    """
    def __init__(self, depth, channels):
        super().__init__()
        self.depth = depth
        self.channels = channels
        
        # Core processing
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        
        # Pointwise with SE
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.se = SEBlock(channels, reduction=16)
        self.bn2 = nn.BatchNorm2d(channels)
        
        if depth > 1:
            # Recursive branches
            self.left = FractalDWBlock(depth - 1, channels)
            self.right = FractalDWBlock(depth - 1, channels)
            # Simple averaging for stability (as in Model 1)
            # No gating to avoid optimization issues in 1 epoch

    def forward(self, x):
        if self.depth > 1:
            left_out = self.left(x)
            right_out = self.right(x)
            # Average the parallel branches
            x = (left_out + right_out) / 2.0
        
        # Core
        y = self.dw_conv(x)
        y = self.bn1(y)
        y = self.act(y)
        
        y = self.pw_conv(y)
        y = self.se(y)
        y = self.bn2(y)
        
        # Residual connection
        return x + y

class MultiScaleStem(nn.Module):
    """
    Stem with parallel 3x3 and 5x5 convolutions to capture multi-scale features.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Branch 1: 3x3
        self.conv3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels // 2)
        
        # Branch 2: 5x5
        self.conv5 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=5, padding=2, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels // 2)
        
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x3 = self.bn3(self.conv3(x))
        x5 = self.bn5(self.conv5(x))
        x = torch.cat([x3, x5], dim=1)
        return self.act(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Non-standard channel progression: 48 -> 96 -> 192 -> 384
        # Avoids 64-128-256-512
        
        # Stem: Multi-scale 3->48
        self.stem = MultiScaleStem(in_ch, 48)
        
        # Stage 1: 48 channels, 32x32 -> 32x32
        # Depth 2 Fractal
        self.stage1 = FractalDWBlock(depth=2, channels=48)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition 1: 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 96 channels, 16x16 -> 16x16
        # Depth 3 Fractal
        self.stage2 = FractalDWBlock(depth=3, channels=96)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition 2: 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 192 channels, 8x8 -> 8x8
        # Depth 3 Fractal
        self.stage3 = FractalDWBlock(depth=3, channels=192)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition 3: 192 -> 384
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        # Stage 4: 384 channels, 4x4 -> 4x4
        # Depth 2 Fractal
        self.stage4 = FractalDWBlock(depth=2, channels=384)
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(384, n_classes)
        
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
        # Assume harness applies transform (norm/flip)
        # Do NOT normalize here to avoid double normalization
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
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Model 1 success: AdamW, lr=0.002, beta1=0.75
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Estimate total steps for 50,000 CIFAR-10 images
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        # Cosine Annealing for stable 1-epoch convergence
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=prm['lr'] * 0.01
        )
        
        return self.optimizer
    
    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()