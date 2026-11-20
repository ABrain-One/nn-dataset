import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """Stochastic Depth (DropPath) as used in DeiT/RegNet."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        random_tensor.div_(keep_prob)
        return x * random_tensor

class RegNetBlock(nn.Module):
    """
    A RegNet-style block with an inverted bottleneck (expand -> depthwise -> project).
    Includes a residual connection with a 1x1 projection if channels change.
    """
    def __init__(self, in_ch, out_ch, expand_ratio=4, drop_path_prob=0.0):
        super().__init__()
        self.expand_ratio = expand_ratio
        hidden_dim = int(in_ch * expand_ratio)
        
        # Expansion (1x1)
        self.conv1 = nn.Conv2d(in_ch, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise (3x3)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # Projection (1x1)
        self.conv3 = nn.Conv2d(hidden_dim, out_ch, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        # Activation
        self.act = nn.SiLU(inplace=True)
        
        # Residual projection if channels change
        self.res_proj = nn.Identity()
        if in_ch != out_ch:
            self.res_proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        shortcut = self.res_proj(x)
        
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        
        y = self.conv3(y)
        y = self.bn3(y)
        
        y = self.drop_path(y)
        
        return shortcut + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        
        # Determine input channels robustly
        # in_shape is typically (C, H, W) or (H, W, C) or (N, C, H, W)
        # CIFAR is 3 channels.
        c_in = 3
        if len(in_shape) == 3:
            if in_shape[0] == 3: c_in = 3
            elif in_shape[-1] == 3: c_in = 3
            else: c_in = in_shape[1]
        elif len(in_shape) == 4:
            c_in = in_shape[1]
            
        n_classes = out_shape[0] if out_shape else 10

        # Architecture Design:
        # Goal: Beat 77.14% (Design 1) and 74.34% (Model to Improve).
        # Strategy: Combine the efficiency of Depthwise Separable Convs (Model 1/Design 1)
        # with the "Fractal" parallelism idea (Model 1) but implemented as a stable
        # parallel branch in a single block, and use a deeper, wider RegNet-like structure.
        
        # Channel Progression (Avoiding 64-128-256-512):
        # We will use: 48 -> 96 -> 192 -> 384
        # This is similar to Design 2 but we will use a different block structure (RegNet vs GatedResidual).
        
        # Stem: 3 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, 32x32 -> 16x16
        # 2 blocks
        self.stage1 = nn.Sequential(
            RegNetBlock(48, 48, expand_ratio=4, drop_path_prob=0.0),
            RegNetBlock(48, 48, expand_ratio=4, drop_path_prob=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 channels, 16x16 -> 8x8
        # 3 blocks
        self.stage2 = nn.Sequential(
            RegNetBlock(48, 96, expand_ratio=4, drop_path_prob=0.0),
            RegNetBlock(96, 96, expand_ratio=4, drop_path_prob=0.0),
            RegNetBlock(96, 96, expand_ratio=4, drop_path_prob=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 192 channels, 8x8 -> 4x4
        # 4 blocks
        self.stage3 = nn.Sequential(
            RegNetBlock(96, 192, expand_ratio=4, drop_path_prob=0.0),
            RegNetBlock(192, 192, expand_ratio=4, drop_path_prob=0.0),
            RegNetBlock(192, 192, expand_ratio=4, drop_path_prob=0.0),
            RegNetBlock(192, 192, expand_ratio=4, drop_path_prob=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 384 channels, 4x4
        # 2 blocks
        self.stage4 = nn.Sequential(
            RegNetBlock(192, 384, expand_ratio=4, drop_path_prob=0.0),
            RegNetBlock(384, 384, expand_ratio=4, drop_path_prob=0.0)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(384, n_classes)
        
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
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Fixed training setup: AdamW, cosine schedule, batch 128
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
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            
            self.optimizer.step()
            scheduler.step()