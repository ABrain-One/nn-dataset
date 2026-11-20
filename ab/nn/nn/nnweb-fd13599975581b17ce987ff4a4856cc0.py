import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        # x: B, C, H, W
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y

class HierDWBlock(nn.Module):
    """
    Hierarchical Depthwise Block.
    Expands channels, applies a sequence of Depthwise Convs (depth 'dw_depth'),
    applies SE gating, projects back, and adds residual.
    """
    def __init__(self, in_channels, out_channels, dw_depth=2, expand_ratio=2):
        super(HierDWBlock, self).__init__()
        expanded_channels = in_channels * expand_ratio
        
        # Expand
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded_channels)
        
        # Recursive/Sequential Depthwise Convs
        dw_convs = []
        for _ in range(dw_depth):
            dw_convs.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, padding=1, groups=expanded_channels, bias=False))
            dw_convs.append(nn.BatchNorm2d(expanded_channels))
            dw_convs.append(nn.SiLU(inplace=True))
        self.dw_seq = nn.Sequential(*dw_convs)
        
        # SE Block
        self.se = SEBlock(expanded_channels)
        
        # Project
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Residual shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        
        y = self.expand_conv(x)
        y = self.expand_bn(y)
        y = self.act(y)
        
        y = self.dw_seq(y)
        
        y = self.se(y)
        
        y = self.project_conv(y)
        y = self.project_bn(y)
        
        y = y + shortcut
        y = self.act(y)
        
        return y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 64 channels, spatial 32x32
        # 2 blocks, DW depth 2
        self.stage1 = nn.Sequential(
            HierDWBlock(64, 64, dw_depth=2, expand_ratio=2),
            HierDWBlock(64, 64, dw_depth=2, expand_ratio=2)
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Stage 2: 128 channels, spatial 16x16
        # 2 blocks, DW depth 3
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            HierDWBlock(128, 128, dw_depth=3, expand_ratio=2),
            HierDWBlock(128, 128, dw_depth=3, expand_ratio=2)
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Stage 3: 256 channels, spatial 8x8
        # 2 blocks, DW depth 4
        self.trans2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            HierDWBlock(256, 256, dw_depth=4, expand_ratio=2),
            HierDWBlock(256, 256, dw_depth=4, expand_ratio=2)
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
        
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Model 1 used lr=0.002, momentum=0.75 (beta1)
        # We use a cosine schedule with warmup for better 1-epoch convergence.
        
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
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
            
        # Warmup for first 5% of steps, then Cosine decay
        warmup_steps = max(1, int(0.05 * total_steps))
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay from 1.0 to 0.0
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
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