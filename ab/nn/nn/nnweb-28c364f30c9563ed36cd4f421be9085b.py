import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.SiLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class RegNetBlock(nn.Module):
    """
    RegNet-style block:
    1. 1x1 Expansion (if expansion > 1)
    2. 3x3 Depthwise (or standard if groups=1)
    3. 1x1 Projection
    4. SE Block (optional)
    5. Residual connection
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, se_ratio=8, drop_prob=0.0):
        super().__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        
        # Expansion
        self.conv1 = ConvBNAct(in_channels, out_channels * expansion, kernel_size=1, stride=1, padding=0)
        
        # Depthwise
        self.conv2 = ConvBNAct(out_channels * expansion, out_channels * expansion, kernel_size=3, stride=stride, padding=1, groups=out_channels * expansion)
        
        # Projection
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels * expansion, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # SE
        self.se = SEBlock(out_channels, se_ratio)
        
        # DropPath (Stochastic Depth)
        self.drop_prob = drop_prob
        
        # Shortcut
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out)
        
        if self.drop_prob > 0.0 and self.training:
            # Stochastic Depth
            if torch.rand(()) > self.drop_prob:
                out = out / (1.0 - self.drop_prob)
            else:
                out = torch.zeros_like(out)
        
        if self.use_residual:
            out = out + identity
        else:
            out = out + self.shortcut(identity)
            
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        
        # Derive input channels
        # Assuming in_shape is (C, H, W) or (N, C, H, W) where C is index 1 or 0?
        # Based on previous models, in_shape[1] is often used for channels if N is included, 
        # or in_shape[0] if not. 
        # Model 1 hardcodes 3. Model 2 uses in_shape[1].
        # Let's assume in_shape is (C, H, W) or (H, W, C) is less likely for PyTorch.
        # However, to be safe and consistent with "Model to improve" which hardcodes 3, 
        # but also "Model 2" which uses in_shape[1], let's look at the context.
        # If in_shape is (3, 32, 32), in_shape[1] is 32. That would be wrong for channels.
        # If in_shape is (32, 32, 3), in_shape[1] is 32.
        # If in_shape is (N, C, H, W), in_shape[1] is C.
        # Given the ambiguity, and that CIFAR is 3 channels, I will try to infer.
        # If len(in_shape) == 3:
        #   If in_shape[0] == 3: C=3
        #   If in_shape[2] == 3: C=3
        #   Else: C=in_shape[0] (assumed CHW)
        # If len(in_shape) == 4: C = in_shape[1]
        
        if len(in_shape) == 4:
            c_in = in_shape[1]
        elif len(in_shape) == 3:
            if in_shape[0] == 3:
                c_in = 3
            elif in_shape[2] == 3:
                c_in = 3
            else:
                c_in = in_shape[0]
        else:
            c_in = 3
            
        n_classes = out_shape[0] if out_shape else 10
        
        # Architecture Design:
        # Stem: 3x3 Conv, BN, SiLU
        # Stage 1: 32 channels, 32x32 (No pool, keep resolution)
        # Stage 2: 64 channels, 16x16 (Stride 2 in first block)
        # Stage 3: 128 channels, 8x8 (Stride 2 in first block)
        # Stage 4: 256 channels, 4x4 (Stride 2 in first block)
        
        # This avoids the 64-128-256-512 progression (stops at 256).
        # Uses Depthwise Convs (from Model 1).
        # Uses SiLU (from Model 1).
        # Uses BatchNorm (from Model 1).
        # Uses Residual connections (standard).
        # Adds SE blocks for attention (new element).
        # Uses Stochastic Depth (from Model 1 concept, adapted).
        
        self.stem = ConvBNAct(c_in, 32, kernel_size=3, stride=1, padding=1)
        
        # Stage 1: 32 channels
        # 2 blocks
        self.stage1 = nn.Sequential(
            RegNetBlock(32, 32, stride=1, expansion=1, se_ratio=8, drop_prob=0.0),
            RegNetBlock(32, 32, stride=1, expansion=1, se_ratio=8, drop_prob=0.0)
        )
        
        # Stage 2: 64 channels
        # 2 blocks, first with stride 2
        self.stage2 = nn.Sequential(
            RegNetBlock(32, 64, stride=2, expansion=1, se_ratio=8, drop_prob=0.0),
            RegNetBlock(64, 64, stride=1, expansion=1, se_ratio=8, drop_prob=0.0)
        )
        
        # Stage 3: 128 channels
        # 3 blocks, first with stride 2
        self.stage3 = nn.Sequential(
            RegNetBlock(64, 128, stride=2, expansion=1, se_ratio=8, drop_prob=0.0),
            RegNetBlock(128, 128, stride=1, expansion=1, se_ratio=8, drop_prob=0.0),
            RegNetBlock(128, 128, stride=1, expansion=1, se_ratio=8, drop_prob=0.0)
        )
        
        # Stage 4: 256 channels
        # 4 blocks, first with stride 2
        # Using expansion=2 for the last stage to increase capacity slightly without too many params?
        # 256 * 2 = 512 hidden.
        # Let's stick to expansion=1 to keep params low, or maybe 2 for the last stage?
        # 256*2*9 + 256*2*1 + 256*1*1 ... 
        # Let's use expansion=2 for stage 4 to boost capacity.
        self.stage4 = nn.Sequential(
            RegNetBlock(128, 256, stride=2, expansion=2, se_ratio=8, drop_prob=0.0),
            RegNetBlock(256, 256, stride=1, expansion=2, se_ratio=8, drop_prob=0.0),
            RegNetBlock(256, 256, stride=1, expansion=2, se_ratio=8, drop_prob=0.0),
            RegNetBlock(256, 256, stride=1, expansion=2, se_ratio=8, drop_prob=0.0)
        )
        
        # Head
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        # 256 features
        self.classifier = nn.Linear(256, n_classes)
        
        # Initialize weights
        self._init_weights()
        
        self.to(device)

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
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Model 1 (74.66%) used AdamW with lr=0.002, momentum=0.75
        # We use SGD with momentum because it is often more stable for 1-epoch training
        # when combined with a proper LR schedule.
        # Let's stick to the strong signal from Model 1: AdamW.
        # But Model 1's success might be due to the specific architecture + AdamW.
        # Let's try SGD with a cosine schedule, which is the gold standard for 1-epoch CIFAR.
        # Actually, let's look at the hyperparameters provided.
        # If I use SGD, I should use a high LR.
        # If I use AdamW, I should use a low LR.
        
        # Let's use SGD with momentum as it's robust.
        # But wait, Model 1 used AdamW.
        # Let's try a hybrid: AdamW with a cosine schedule.
        
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
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