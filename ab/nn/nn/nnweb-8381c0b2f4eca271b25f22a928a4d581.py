import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2, se_reduction=4):
        super().__init__()
        expanded = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # 1. Expand
        self.conv1 = nn.Conv2d(in_channels, expanded, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded)
        
        # 2. Depthwise
        self.conv2 = nn.Conv2d(expanded, expanded, kernel_size=3, stride=stride, padding=1, groups=expanded, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded)
        
        # 3. SE
        self.se = SEBlock(expanded, se_reduction)
        
        # 4. Project
        self.conv3 = nn.Conv2d(expanded, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        
        x = self.se(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.use_residual:
            x = x + input
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        self.batch_size = prm.get('batch', 128)
        
        in_ch = in_shape[1]
        num_classes = out_shape[0]
        
        # Stem: 3 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 -> 48 (2 blocks), stride 1
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(48, 48, stride=1, expand_ratio=2),
            DepthwiseSeparableBlock(48, 48, stride=1, expand_ratio=2)
        )
        
        # Transition: 48 -> 96, stride 2
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 -> 96 (3 blocks), stride 1
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(96, 96, stride=1, expand_ratio=2),
            DepthwiseSeparableBlock(96, 96, stride=1, expand_ratio=2),
            DepthwiseSeparableBlock(96, 96, stride=1, expand_ratio=2)
        )
        
        # Transition: 96 -> 192, stride 2
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 192 -> 192 (4 blocks), stride 1
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(192, 192, stride=1, expand_ratio=2),
            DepthwiseSeparableBlock(192, 192, stride=1, expand_ratio=2),
            DepthwiseSeparableBlock(192, 192, stride=1, expand_ratio=2),
            DepthwiseSeparableBlock(192, 192, stride=1, expand_ratio=2)
        )
        
        # Transition: 192 -> 384, stride 2
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 384 -> 384 (2 blocks), stride 1
        self.stage4 = nn.Sequential(
            DepthwiseSeparableBlock(384, 384, stride=1, expand_ratio=2),
            DepthwiseSeparableBlock(384, 384, stride=1, expand_ratio=2)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(384, num_classes)
        
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
        # Apply normalization if not done externally
        # The 'transform' hyperparameter indicates if the data is already normalized or if we should do it here.
        # Standard CIFAR-10 mean/std: 0.4914, 0.2470
        if self.transform in ['norm_32', 'norm_32_flip']:
             # If the data is raw [0,1], we normalize. 
             # However, usually 'norm_32' implies the data is already normalized or we normalize it here.
             # To be safe, if the input mean is around 0.5, we normalize.
             # But the prompt says "transform" is a hyperparameter. 
             # Let's assume the data is raw [0,1] and we normalize here if transform is specified.
             # If the data is already normalized, this would be wrong. 
             # Given the ambiguity, and that Model 1 didn't normalize in forward, 
             # I will assume the data is already normalized if 'norm_32' is passed, 
             # OR I will normalize here. 
             # Let's look at Design 3: it normalized in forward. 
             # Let's look at Model 1: it did NOT normalize in forward. 
             # I will NOT normalize in forward to match Model 1's behavior, 
             # assuming the DataLoader provides normalized data.
             pass
        
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
        
        # Use AdamW as in Model 1, which was the strongest.
        # Momentum is passed as 'momentum'. In AdamW, this maps to beta1.
        # Model 1 used momentum=0.75.
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=prm['lr'], 
            betas=(prm['momentum'], 0.999), 
            weight_decay=1e-4
        )
        
        # Estimate total steps for scheduler
        # CIFAR-10 has 50,000 images.
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