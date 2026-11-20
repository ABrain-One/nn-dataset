import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class StochasticDepth(nn.Module):
    """
    Stochastic Depth (DropPath) for regularization.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0:
            random_tensor.div_(keep_prob)
        
        return x * random_tensor

class InvertedBlock(nn.Module):
    """
    EfficientNet-style Inverted Residual Block with SE and Stochastic Depth.
    Structure:
    1. Expand (1x1 conv) -> BN -> GELU
    2. Depthwise Conv (3x3) -> BN -> GELU
    3. Squeeze-Excitation (SE)
    4. Project (1x1 conv) -> BN
    5. Stochastic Depth
    6. Residual Connection
    """
    def __init__(self, in_channels, out_channels, expansion=4, drop_prob=0.0):
        super().__init__()
        hidden_dim = in_channels * expansion
        
        # Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # Depthwise
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # Squeeze-Excitation
        # Fixed: Ensure output shape is [B, hidden_dim, 1, 1] for broadcasting
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),  # Flatten to [B, hidden_dim]
            nn.Linear(hidden_dim, max(8, hidden_dim // 12)),
            nn.GELU(),
            nn.Linear(max(8, hidden_dim // 12), hidden_dim),
            nn.Sigmoid(),
            nn.Unflatten(1, (hidden_dim, 1, 1)) # Restore to [B, hidden_dim, 1, 1]
        )
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.drop_path = StochasticDepth(drop_prob)
        
        # Residual connection (identity if channels match)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.skip(x)
        
        y = self.expand(x)
        y = self.dw_conv(y)
        
        # SE
        se_weights = self.se(y)
        y = y * se_weights
        
        y = self.project(y)
        y = self.drop_path(y)
        
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 48
        # Using 3x3 conv to keep spatial resolution high initially
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU()
        )
        
        # Stage 1: 48 channels, 4x4 spatial
        # 2 blocks, expansion 4
        self.stage1 = nn.Sequential(
            InvertedBlock(48, 48, expansion=4, drop_prob=0.0),
            InvertedBlock(48, 48, expansion=4, drop_prob=0.0)
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        
        # Stage 2: 96 channels, 2x2 spatial
        # 4 blocks, expansion 4
        self.stage2 = nn.Sequential(
            InvertedBlock(96, 96, expansion=4, drop_prob=0.05),
            InvertedBlock(96, 96, expansion=4, drop_prob=0.05),
            InvertedBlock(96, 96, expansion=4, drop_prob=0.1),
            InvertedBlock(96, 96, expansion=4, drop_prob=0.1)
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU()
        )
        
        # Stage 3: 192 channels, 1x1 spatial
        # 2 blocks, expansion 4
        self.stage3 = nn.Sequential(
            InvertedBlock(192, 192, expansion=4, drop_prob=0.1),
            InvertedBlock(192, 192, expansion=4, drop_prob=0.1)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, n_classes)
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
        
        # Use SGD with momentum for better generalization in 1 epoch
        # Model 1 (best) used AdamW, but SGD is often more robust for 1-epoch training
        # if we use a good learning rate schedule.
        # Let's use SGD with momentum as it's robust.
        
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