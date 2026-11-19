import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class AttentionBlock(nn.Module):
    """
    Lightweight Channel Attention (SE-like) + Spatial Attention (CA-like)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        
        # Channel Attention
        self.channel_fc1 = nn.Conv2d(channels, mid, 1)
        self.channel_fc2 = nn.Conv2d(mid, channels, 1)
        
        # Spatial Attention
        self.spatial_fc1 = nn.Conv2d(2, 1, 7, padding=3)
        
    def forward(self, x):
        # Channel
        avg = F.adaptive_avg_pool2d(x, 1)
        maxv = F.adaptive_max_pool2d(x, 1)
        y = avg + maxv
        y = F.silu(self.channel_fc1(y), inplace=True)
        y = torch.sigmoid(self.channel_fc2(y))
        x = x * y
        
        # Spatial
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True)[0]
        s = torch.cat([avg_s, max_s], dim=1)
        s = torch.sigmoid(self.spatial_fc1(s))
        x = x * s
        return x

class EfficientBlock(nn.Module):
    """
    Depthwise Separable Convolution with optional Attention
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        expanded = in_channels * expand_ratio
        
        # Expand
        self.conv1 = nn.Conv2d(in_channels, expanded, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded)
        
        # Depthwise
        self.conv2 = nn.Conv2d(expanded, expanded, 3, stride=stride, padding=1, groups=expanded, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded)
        
        # Project
        self.conv3 = nn.Conv2d(expanded, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True)
        
        if use_attention:
            self.att = AttentionBlock(expanded)
        
        # Skip connection projection if needed
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        
        if self.use_attention:
            y = self.att(y)
            
        y = self.conv3(y)
        y = self.bn3(y)
        
        if self.skip is not None:
            identity = self.skip(identity)
            
        y = y + identity
        return y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Reduced channel widths to stay under 5M parameters
        # Stem: 3 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 channels, 32x32 -> 32x32
        self.stage1 = nn.Sequential(
            EfficientBlock(32, 32, stride=1, expand_ratio=4, use_attention=False),
            EfficientBlock(32, 32, stride=1, expand_ratio=4, use_attention=False)
        )
        
        # Transition: 32 -> 64, 32x32 -> 16x16
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 64 channels, 16x16
        self.stage2 = nn.Sequential(
            EfficientBlock(64, 64, stride=1, expand_ratio=4, use_attention=False),
            EfficientBlock(64, 64, stride=1, expand_ratio=4, use_attention=False),
            EfficientBlock(64, 64, stride=1, expand_ratio=4, use_attention=False)
        )
        
        # Transition: 64 -> 128, 16x16 -> 8x8
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 128 channels, 8x8
        # Attention enabled here to boost accuracy
        self.stage3 = nn.Sequential(
            EfficientBlock(128, 128, stride=1, expand_ratio=4, use_attention=True),
            EfficientBlock(128, 128, stride=1, expand_ratio=4, use_attention=True),
            EfficientBlock(128, 128, stride=1, expand_ratio=4, use_attention=True)
        )
        
        # Transition: 128 -> 256, 8x8 -> 4x4
        self.trans3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 256 channels, 4x4
        # Attention enabled here
        self.stage4 = nn.Sequential(
            EfficientBlock(256, 256, stride=1, expand_ratio=4, use_attention=True),
            EfficientBlock(256, 256, stride=1, expand_ratio=4, use_attention=True)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)
        
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
        
        # SGD with momentum
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
        
        # Estimate total steps for scheduler
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        # Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps, 
            eta_min=prm['lr'] * 0.01
        )
        
        return self.optimizer
    
    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
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