import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class DepthwiseInvertedBlock(nn.Module):
    """
    MobileNet-style inverted residual block.
    Structure: 1x1 Conv (expand) -> BN -> SiLU -> 3x3 DW Conv -> BN -> SiLU -> 1x1 Conv (project) -> BN
    Residual connection if input/output channels match.
    """
    def __init__(self, in_channels, out_channels, expand_ratio=4):
        super().__init__()
        self.use_residual = (in_channels == out_channels)
        
        expanded = in_channels * expand_ratio
        self.conv1 = nn.Conv2d(in_channels, expanded, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded)
        self.act1 = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv2d(expanded, expanded, kernel_size=3, padding=1, groups=expanded, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded)
        self.act2 = nn.SiLU(inplace=True)
        
        self.conv3 = nn.Conv2d(expanded, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.use_residual:
            out += identity
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Reduced channel widths to fit under 5M parameters
        # Target: ~3.5M - 4.5M parameters
        
        # Stem: 3 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 -> 32 (1 block)
        self.stage1 = nn.Sequential(
            DepthwiseInvertedBlock(32, 32, expand_ratio=4)
        )
        
        # Pool 1
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Stage 2: 32 -> 64 (2 blocks)
        self.stage2 = nn.Sequential(
            DepthwiseInvertedBlock(32, 64, expand_ratio=4),
            DepthwiseInvertedBlock(64, 64, expand_ratio=4)
        )
        
        # Pool 2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Stage 3: 64 -> 128 (3 blocks)
        self.stage3 = nn.Sequential(
            DepthwiseInvertedBlock(64, 128, expand_ratio=4),
            DepthwiseInvertedBlock(128, 128, expand_ratio=4),
            DepthwiseInvertedBlock(128, 128, expand_ratio=4)
        )
        
        # Pool 3
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Stage 4: 128 -> 256 (2 blocks)
        self.stage4 = nn.Sequential(
            DepthwiseInvertedBlock(128, 256, expand_ratio=4),
            DepthwiseInvertedBlock(256, 256, expand_ratio=4)
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
        
        # Use SGD with momentum for stability in 1-epoch training
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
        # Start at lr, end at 0
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