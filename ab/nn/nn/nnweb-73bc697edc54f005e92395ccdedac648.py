import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, max(channels // reduction, 16))
        self.fc2 = nn.Linear(max(channels // reduction, 16), channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        x_avg = self.avg_pool(x).view(b, c)
        x_avg = self.act(self.fc1(x_avg))
        x_avg = self.fc2(x_avg)
        x_avg = torch.sigmoid(x_avg).view(b, c, 1, 1)
        return x * x_avg

class EfficientBlock(nn.Module):
    """
    Inverted Bottleneck Block:
    1x1 Expand -> BN -> SiLU -> 3x3 DW -> BN -> SqueezeExcite -> 1x1 Project -> BN
    """
    def __init__(self, in_channels, out_channels, expand_ratio=4, use_se=True):
        super().__init__()
        self.use_se = use_se
        self.expand_ratio = expand_ratio
        hidden_dim = int(in_channels * expand_ratio)
        
        # Expansion
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.SiLU(inplace=True)
        
        # Depthwise
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = nn.SiLU(inplace=True)
        
        # Squeeze and Excite
        if use_se:
            self.se = SqueezeExcite(hidden_dim, reduction=4)
        else:
            self.se = nn.Identity()
            
        # Projection
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        out = self.se(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 32 (Small start to save params, efficient for 1 epoch)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 -> 32, 2 blocks, no pool (keep 32x32)
        self.stage1 = nn.Sequential(
            EfficientBlock(32, 32, expand_ratio=4, use_se=False),
            EfficientBlock(32, 32, expand_ratio=4, use_se=False),
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Stage 2: 32 -> 64, 2 blocks, pool to 16x16
        self.stage2 = nn.Sequential(
            EfficientBlock(32, 64, expand_ratio=4, use_se=False),
            EfficientBlock(64, 64, expand_ratio=4, use_se=False),
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Stage 3: 64 -> 128, 2 blocks, pool to 8x8
        self.stage3 = nn.Sequential(
            EfficientBlock(64, 128, expand_ratio=4, use_se=True),
            EfficientBlock(128, 128, expand_ratio=4, use_se=True),
        )
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Stage 4: 128 -> 256, 2 blocks, keep 8x8 (no pool)
        self.stage4 = nn.Sequential(
            EfficientBlock(128, 256, expand_ratio=4, use_se=True),
            EfficientBlock(256, 256, expand_ratio=4, use_se=True),
        )
        
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
        # CIFAR-10 mean/std
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with momentum. 
        # Model 1 (74.66%) used AdamW. Model 2 (69%) used SGD.
        # My previous designs with AdamW got ~72-73%.
        # SGD with momentum 0.9 and higher LR often generalizes better in 1 epoch if the architecture is good.
        # Let's try SGD.
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
        
        # Cosine schedule
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