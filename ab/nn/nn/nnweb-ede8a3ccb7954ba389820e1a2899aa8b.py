import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class InvertedBlock(nn.Module):
    """
    EfficientNet-style Inverted Residual Block.
    1x1 Conv -> BN -> SiLU -> 3x3 Depthwise Conv -> BN -> 1x1 Conv -> BN
    Includes SE block operating on channel dimension.
    """
    def __init__(self, in_channels, out_channels, expand_ratio=4, use_residual=True):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        
        hidden = in_channels * expand_ratio
        
        # Expansion
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True)
        )
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True)
        )
        
        # Projection
        self.projection = nn.Sequential(
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Squeeze-and-Excitation (SE) block
        # Correctly handles [B, C, H, W] -> [B, C] -> [B, C, 1, 1]
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # [B, C, 1, 1]
            nn.Flatten(1),               # [B, C]
            nn.Linear(out_channels, max(1, out_channels // 8)),
            nn.SiLU(inplace=True),
            nn.Linear(max(1, out_channels // 8), out_channels),
            nn.Sigmoid(),
            nn.Unflatten(1, (out_channels, 1, 1)) # [B, C, 1, 1]
        )
        
        # Residual connection
        if self.use_residual:
            self.shortcut = nn.Identity()
        else:
            # If shapes don't match, we skip residual addition in forward logic
            # We handle this by checking use_residual in forward
            pass
            
    def forward(self, x):
        identity = x
        
        out = self.expansion(x)
        out = self.depthwise(out)
        out = self.projection(out)
        
        # SE
        se_weights = self.se(out)
        out = out * se_weights
        
        if self.use_residual:
            out = out + identity
            
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 48
        # Using a slightly wider start to capture more features early
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 -> 96, 2 blocks
        self.stage1 = nn.Sequential(
            InvertedBlock(48, 48, expand_ratio=4, use_residual=True),
            InvertedBlock(48, 96, expand_ratio=4, use_residual=False), # Channel change, no residual
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 -> 192, 3 blocks
        self.stage2 = nn.Sequential(
            InvertedBlock(96, 96, expand_ratio=4, use_residual=True),
            InvertedBlock(96, 96, expand_ratio=4, use_residual=True),
            InvertedBlock(96, 192, expand_ratio=4, use_residual=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 192 -> 384, 3 blocks
        self.stage3 = nn.Sequential(
            InvertedBlock(192, 192, expand_ratio=4, use_residual=True),
            InvertedBlock(192, 192, expand_ratio=4, use_residual=True),
            InvertedBlock(192, 384, expand_ratio=4, use_residual=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Head
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
        # Apply normalization if needed
        # CIFAR-10 mean/std
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with momentum, which is robust for 1-epoch training
        # Combined with a cosine LR schedule
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
        
        # Store batch size
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