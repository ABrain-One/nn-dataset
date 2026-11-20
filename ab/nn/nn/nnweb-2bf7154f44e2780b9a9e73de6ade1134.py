import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class MBConvBlock(nn.Module):
    """
    MobileNetV2 style block with residual connection.
    Expand -> Depthwise -> Project
    """
    def __init__(self, in_ch, out_ch, expand_ratio, stride=1):
        super().__init__()
        expanded = in_ch * expand_ratio
        
        # Expansion (1x1)
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, expanded, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.SiLU(inplace=True)
        )
        
        # Depthwise
        self.dw = nn.Sequential(
            nn.Conv2d(expanded, expanded, kernel_size=3, stride=stride, padding=1, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.SiLU(inplace=True)
        )
        
        # Projection (1x1)
        self.project = nn.Sequential(
            nn.Conv2d(expanded, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        
        # Residual connection (only if shapes match)
        self.residual = (stride == 1 and in_ch == out_ch)
        
    def forward(self, x):
        shortcut = x
        out = self.expand(x)
        out = self.dw(out)
        out = self.project(out)
        if self.residual:
            out = out + shortcut
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 48
        # No pooling here, stride 1 to keep resolution high initially
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Architecture:
        # Stage 1: 48 ch, 1 block, stride 1
        # Stage 2: 96 ch, 2 blocks, stride 2 (first block), 1 (second)
        # Stage 3: 192 ch, 3 blocks, stride 2 (first block), 1 (others)
        # Stage 4: 384 ch, 1 block, stride 1
        
        # Note: Using expand_ratio=1 for early stages to save params, 2 for later?
        # Let's use expand_ratio=1 for all to keep it lightweight and fast, relying on depth.
        # Actually, MBConv with ratio 1 is just DW+1x1+1x1.
        
        # Stage 1: 48 -> 48
        self.stage1 = nn.Sequential(
            MBConvBlock(48, 48, expand_ratio=1, stride=1)
        )
        
        # Stage 2: 48 -> 96 (stride 2)
        self.stage2 = nn.Sequential(
            MBConvBlock(48, 96, expand_ratio=1, stride=2),
            MBConvBlock(96, 96, expand_ratio=1, stride=1)
        )
        
        # Stage 3: 96 -> 192 (stride 2)
        self.stage3 = nn.Sequential(
            MBConvBlock(96, 192, expand_ratio=1, stride=2),
            MBConvBlock(192, 192, expand_ratio=1, stride=1),
            MBConvBlock(192, 192, expand_ratio=1, stride=1)
        )
        
        # Stage 4: 192 -> 384 (stride 1)
        self.stage4 = nn.Sequential(
            MBConvBlock(192, 384, expand_ratio=1, stride=1),
            MBConvBlock(384, 384, expand_ratio=1, stride=1)
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
        x = self.stage4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # SGD with momentum is often superior for 1-epoch training on CIFAR
        # when paired with a proper schedule.
        # Model 1 (74.66%) used AdamW.
        # Model 3 (67.63%) used SGD.
        # Let's try SGD with a high LR and cosine decay.
        # Momentum is passed in prm.
        
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