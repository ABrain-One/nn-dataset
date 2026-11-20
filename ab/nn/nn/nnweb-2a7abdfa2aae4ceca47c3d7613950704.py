import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style attention.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    """
    Simple spatial attention (SE-like spatial variant).
    Fixed to handle arbitrary channel counts by using AdaptiveAvgPool2d to reduce spatial dimensions
    to 1x1 before the 1x1 convolution, or by using a Conv2d that accepts the input channels.
    Here we use a 1x1 conv to project the concatenated [avg, max] (2C channels) to C channels.
    """
    def __init__(self, channels):
        super().__init__()
        # Input to conv is 2*channels (concat of avg and max)
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is [B, C, H, W]
        avg_out = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        max_out = F.adaptive_max_pool2d(x, 1)   # [B, C, 1, 1]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2C, 1, 1]
        attention = self.sigmoid(self.conv(combined))  # [B, C, 1, 1]
        return x * attention

class EfficientBlock(nn.Module):
    """
    Inverted Residual Block:
    1x1 Expand -> BN -> SiLU -> 3x3 DW -> BN -> SiLU -> 1x1 Project -> BN
    + SE Attention
    + Residual Connection
    """
    def __init__(self, in_channels, out_channels, expand_ratio=4, stride=1, use_se=True):
        super().__init__()
        self.use_se = use_se
        self.residual = (in_channels == out_channels and stride == 1)
        
        expanded = in_channels * expand_ratio
        
        self.conv1 = nn.Conv2d(in_channels, expanded, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded)
        self.act1 = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv2d(expanded, expanded, kernel_size=3, padding=1, groups=expanded, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded)
        self.act2 = nn.SiLU(inplace=True)
        
        self.conv3 = nn.Conv2d(expanded, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        if use_se:
            self.se = ChannelAttention(expanded)
        else:
            self.se = nn.Identity()

    def forward(self, x):
        shortcut = x
        
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)
        
        y = self.se(y)
        
        y = self.conv3(y)
        y = self.bn3(y)
        
        if self.residual:
            y = y + shortcut
        return y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 channels, 2 blocks
        self.stage1 = nn.Sequential(
            EfficientBlock(32, 32, expand_ratio=4, stride=1, use_se=False),
            EfficientBlock(32, 32, expand_ratio=4, stride=1, use_se=False)
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 32 -> 64
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 64 channels, 3 blocks
        self.stage2 = nn.Sequential(
            EfficientBlock(64, 64, expand_ratio=4, stride=1, use_se=True),
            EfficientBlock(64, 64, expand_ratio=4, stride=1, use_se=True),
            EfficientBlock(64, 64, expand_ratio=4, stride=1, use_se=True)
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 64 -> 128
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 128 channels, 4 blocks
        self.stage3 = nn.Sequential(
            EfficientBlock(128, 128, expand_ratio=4, stride=1, use_se=True),
            EfficientBlock(128, 128, expand_ratio=4, stride=1, use_se=True),
            EfficientBlock(128, 128, expand_ratio=4, stride=1, use_se=True),
            EfficientBlock(128, 128, expand_ratio=4, stride=1, use_se=True)
        )
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 128 -> 256
        self.trans3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # Stage 4: 256 channels, 2 blocks
        self.stage4 = nn.Sequential(
            EfficientBlock(256, 256, expand_ratio=4, stride=1, use_se=True),
            EfficientBlock(256, 256, expand_ratio=4, stride=1, use_se=True)
        )
        
        # Head with Spatial Attention
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize SpatialAttention with the correct number of channels (256)
        self.spatial_attn = SpatialAttention(channels=256)
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
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.trans3(x)
        x = self.stage4(x)
        
        # Apply spatial attention before global pooling
        x = self.spatial_attn(x)
        
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with momentum for stability in 1-epoch training
        # Model 1 (74.66%) used AdamW, but SGD is often more robust for CIFAR-10
        # with proper LR scheduling.
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
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