import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'transform', 'batch'}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        mid = max(1, channels // reduction)
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

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(GhostModule, self).__init__()
        self.cheap_conv = nn.Conv2d(in_channels, out_channels // ratio, kernel_size,
                                     stride=stride, padding=kernel_size // 2, bias=False)
        self.ghost_conv = nn.Conv2d(out_channels // ratio, out_channels // ratio, dw_size,
                                     dw_size // 2, dw_size // 2, groups=out_channels // ratio, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.cheap_conv(x)
        x2 = self.ghost_conv(x1)
        out = torch.cat([x1, x2], 1)
        out = self.bn(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        c_in = in_shape[1]
        n_classes = out_shape[0]
        
        # Reduced channel widths to stay under 5M params
        # 48 -> 96 -> 192 -> 384
        
        # Entry: 32x32 -> 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32x32 -> 16x16, 48 -> 96
        self.stage1_conv = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            SEBlock(96)
        )
        
        # Stage 2: 16x16 -> 16x16, 96 -> 192 (Ghost)
        self.stage2_ghost = nn.Sequential(
            GhostModule(96, 192, kernel_size=3, ratio=2, dw_size=3),
            nn.SiLU(inplace=True),
            SEBlock(192)
        )
        # Skip connection from Stage 1 (96 ch, 16x16) to Stage 2 level (192 ch, 16x16)
        self.skip1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 16x16 -> 8x8, 192 -> 384
        self.stage3_conv = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True),
            SEBlock(384)
        )
        # Skip connection from Stage 2 (192 ch, 16x16) to Stage 3 level (384 ch, 8x8)
        self.skip2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        # Fusion: 384 (stage3) + 384 (skip2) = 768 -> 384
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True),
            SEBlock(384)
        )
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
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
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Normalize input
        if self.transform in ['norm_32', 'norm_32_flip']:
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(x.device)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1).to(x.device)
            x = (x - mean) / std

        x = self.stem(x)
        
        # Stage 1
        s1 = self.stage1_conv(x)
        
        # Stage 2
        s2_raw = self.stage2_ghost(s1)
        skip1 = self.skip1(s1)
        s2 = s2_raw + skip1
        
        # Stage 3
        s3 = self.stage3_conv(s2)
        skip2 = self.skip2(s2)
        # Resize skip2 to match s3 (8x8)
        skip2 = F.adaptive_avg_pool2d(skip2, s3.shape[2:])
        
        # Fusion
        x = torch.cat([s3, skip2], dim=1)
        x = self.fusion_conv(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        
        # Estimate total steps for scheduler
        batch_size = prm.get('batch', 128)
        # Assume standard CIFAR-10 size 50,000
        total_steps = 50000 // batch_size
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps, 
            eta_min=0.0
        )
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()