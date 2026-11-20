import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FractalDWBlock(nn.Module):
    """
    Fractal-inspired depthwise block with parallel paths.
    Instead of true recursion, we use two parallel depthwise paths
    that are combined via learned gating, then processed through
    a pointwise conv with SE attention.
    """
    def __init__(self, channels, use_se=True):
        super().__init__()
        self.channels = channels
        
        # Two parallel depthwise paths (fractal-inspired parallelism)
        self.dw_path1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.dw_path2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Learned gating to combine parallel paths
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 2, bias=False),
            nn.Softmax(dim=-1)
        )
        
        # Pointwise projection
        self.pw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # SE attention
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x):
        # Process through parallel paths
        p1 = self.dw_path1(x)
        p2 = self.dw_path2(x)
        
        # Compute gate
        gate = self.gate(x)  # [B, 2]
        gate1 = gate[:, 0:1].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        gate2 = gate[:, 1:2].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        
        # Combine parallel paths
        combined = gate1 * p1 + gate2 * p2
        
        # Project and apply attention
        y = self.pw_conv(combined)
        y = self.se(y)
        
        # Residual connection
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        self.batch_size = prm.get('batch', 128)
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, 32x32
        self.stage1 = nn.Sequential(
            FractalDWBlock(48, use_se=False),
            FractalDWBlock(48, use_se=True)
        )
        
        # Transition 1: 48 -> 96, pool to 16x16
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 channels, 16x16
        self.stage2 = nn.Sequential(
            FractalDWBlock(96, use_se=True),
            FractalDWBlock(96, use_se=True),
            FractalDWBlock(96, use_se=True)
        )
        
        # Transition 2: 96 -> 192, pool to 8x8
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 192 channels, 8x8
        self.stage3 = nn.Sequential(
            FractalDWBlock(192, use_se=True),
            FractalDWBlock(192, use_se=True),
            FractalDWBlock(192, use_se=True),
            FractalDWBlock(192, use_se=True)
        )
        
        # Transition 3: 192 -> 384, pool to 4x4
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 384 channels, 4x4
        self.stage4 = nn.Sequential(
            FractalDWBlock(384, use_se=True),
            FractalDWBlock(384, use_se=True)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(384, n_classes)
        
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
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1 (following Model 1's successful approach)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Cosine schedule within one epoch
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=prm['lr'] * 0.01
        )
        
        return self.optimizer
    
    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
        
        for i, (inputs, labels) in enumerate(train_data):
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