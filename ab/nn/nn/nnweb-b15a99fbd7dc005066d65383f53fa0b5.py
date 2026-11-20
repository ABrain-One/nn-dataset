import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=8):
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

class DepthwiseBlock(nn.Module):
    """
    Depthwise separable block with SE attention and residual connection.
    Structure: DW Conv -> BN -> SiLU -> SE -> PW Conv -> BN -> Residual
    """
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        
        # Depthwise convolution
        self.dw_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act = nn.SiLU(inplace=True)
        
        # Squeeze-Excitation
        self.se = SEBlock(in_ch) if use_se else nn.Identity()
        
        # Pointwise convolution
        self.pw_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        # Projection for residual if channels differ
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if (stride != 1 or in_ch != out_ch) else None

    def forward(self, x):
        input = x
        
        # Depthwise
        y = self.dw_conv(x)
        y = self.bn1(y)
        y = self.act(y)
        
        # SE attention
        y = self.se(y)
        
        # Pointwise
        y = self.pw_conv(y)
        y = self.bn2(y)
        
        # Residual
        if self.use_residual:
            y = y + input
        elif self.proj is not None:
            y = y + self.proj(input)
        
        return y

class MultiScaleFusion(nn.Module):
    """
    Lightweight multi-scale feature fusion using dilated convolutions.
    Combines features at different receptive fields.
    """
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        out = self.fuse(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 48, keep spatial size
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, 32x32
        self.stage1 = nn.Sequential(
            DepthwiseBlock(48, 48, stride=1, use_se=False),
            DepthwiseBlock(48, 48, stride=1, use_se=False),
        )
        
        # Transition 1: 48 -> 96, 32x32 -> 16x16
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 channels, 16x16
        self.stage2 = nn.Sequential(
            DepthwiseBlock(96, 96, stride=1, use_se=True),
            DepthwiseBlock(96, 96, stride=1, use_se=True),
            DepthwiseBlock(96, 96, stride=1, use_se=True),
        )
        
        # Multi-scale fusion at stage 2
        self.fusion1 = MultiScaleFusion(96)
        
        # Transition 2: 96 -> 192, 16x16 -> 8x8
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 192 channels, 8x8
        self.stage3 = nn.Sequential(
            DepthwiseBlock(192, 192, stride=1, use_se=True),
            DepthwiseBlock(192, 192, stride=1, use_se=True),
            DepthwiseBlock(192, 192, stride=1, use_se=True),
            DepthwiseBlock(192, 192, stride=1, use_se=True),
        )
        
        # Multi-scale fusion at stage 3
        self.fusion2 = MultiScaleFusion(192)
        
        # Transition 3: 192 -> 384, 8x8 -> 4x4
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 384 channels, 4x4
        self.stage4 = nn.Sequential(
            DepthwiseBlock(384, 384, stride=1, use_se=True),
            DepthwiseBlock(384, 384, stride=1, use_se=True),
        )
        
        # Head: Global average pooling + classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, n_classes)
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
        # Apply normalization if transform suggests it
        # CIFAR-10 mean/std
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.fusion1(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.fusion2(x)
        x = self.trans3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Model 1 (74.66%) used AdamW with lr=0.002, momentum=0.75
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Estimate total steps for 50,000 CIFAR-10 images
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        # Cosine Annealing within one epoch
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()