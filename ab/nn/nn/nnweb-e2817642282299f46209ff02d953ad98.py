import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y

class GhostBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.out_channels = out_channels
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // ratio, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels // ratio),
            nn.SiLU(inplace=True)
        )
        self.ghost_conv = nn.Conv2d(out_channels // ratio, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.cheap_conv(x)
        x = self.ghost_conv(x)
        return x

class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        self.ghost = GhostBlock(out_channels, out_channels, kernel_size=kernel_size, ratio=ratio)
        self.se = SEBlock(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.ghost(x)
        x = self.se(x)
        x = self.act(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        img_size = in_shape[2]

        # Stem: 3 -> 48 (32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 48 -> 48 (32x32)
        self.stage1 = nn.Sequential(
            FractalBlock(48, 48),
            FractalBlock(48, 48)
        )

        # Transition 1: 32x32 -> 16x16, 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )

        # Stage 2: 96 -> 96 (16x16)
        self.stage2 = nn.Sequential(
            FractalBlock(96, 96),
            FractalBlock(96, 96),
            FractalBlock(96, 96)
        )

        # Transition 2: 16x16 -> 8x8, 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )

        # Stage 3: 192 -> 192 (8x8)
        self.stage3 = nn.Sequential(
            FractalBlock(192, 192),
            FractalBlock(192, 192)
        )

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, num_classes)

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
        # Apply normalization inside forward to respect transform
        if self.transform == 'norm_32' or self.transform == 'norm_32_flip':
            x = (x - 0.4914) / 0.2470
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # SGD with momentum and weight decay
        # Use batch size to scale weight decay slightly (heuristic)
        batch_size = prm.get('batch', 128)
        weight_decay = 1e-4 * (batch_size / 128.0)
        
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=weight_decay
        )
        
        # Estimate total steps for scheduler
        # Assuming standard CIFAR-10 (50,000 images)
        total_steps = 50000 // batch_size
        
        # Cosine Annealing for smooth convergence within 1 epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps, 
            eta_min=0
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