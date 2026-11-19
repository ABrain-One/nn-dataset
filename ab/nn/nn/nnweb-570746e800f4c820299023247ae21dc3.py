import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """Stochastic Depth (DropPath)"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        random_tensor.div_(keep_prob)
        return x * random_tensor

class GatedDWBlock(nn.Module):
    """
    Efficient Block with Sigmoid Gating.
    Structure:
    1. 1x1 Conv (Expand) -> BN -> SiLU
    2. 3x3 Depthwise Conv -> BN -> SiLU
    3. Squeeze-Excitation (Gating)
    4. 1x1 Conv (Project) -> BN
    5. Residual Connection + DropPath
    """
    def __init__(self, channels: int, expand_ratio: int = 4, drop_path_prob: float = 0.0, se_ratio: int = 4):
        super().__init__()
        hidden_dim = int(channels * expand_ratio)
        
        # Expansion
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.SiLU(inplace=True)
        
        # Depthwise
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # Squeeze-Excitation (Gating)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(hidden_dim, max(1, hidden_dim // se_ratio)),
            nn.SiLU(inplace=True),
            nn.Linear(max(1, hidden_dim // se_ratio), hidden_dim),
            nn.Sigmoid()
        )
        
        # Projection
        self.conv3 = nn.Conv2d(hidden_dim, channels, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Expand
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        
        # Depthwise
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        
        # Gating (SE)
        se = self.se(y)
        # Reshape SE output to match spatial dimensions of y
        se = se.view(se.size(0), se.size(1), 1, 1)
        y = y * se
        
        # Project
        y = self.conv3(y)
        y = self.bn3(y)
        
        # Residual + DropPath
        y = self.drop_path(y)
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        
        # Determine input channels robustly
        c_in = 3
        if len(in_shape) == 3:
            if in_shape[0] == 3: c_in = 3
            elif in_shape[-1] == 3: c_in = 3
            else: c_in = in_shape[1]
        elif len(in_shape) == 4:
            c_in = in_shape[1]
            
        n_classes = out_shape[0] if out_shape else 10
        
        # Architecture Design:
        # Reduced channel progression to fit under 5M parameters.
        # New progression: 32 -> 64 -> 128 -> 256
        # Reduced block counts: 2, 2, 3, 2
        # Reduced expand ratio to 3 for deeper stages to save params.
        
        # Stem: 3 -> 32, 32x32 -> 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 channels, 32x32 spatial
        # 2 GatedDWBlocks
        self.stage1 = nn.Sequential(
            GatedDWBlock(32, expand_ratio=4, drop_path_prob=0.0),
            GatedDWBlock(32, expand_ratio=4, drop_path_prob=0.0)
        )
        
        # Transition 1: 32 -> 64, 32x32 -> 16x16
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 64 channels, 16x16 spatial
        # 2 GatedDWBlocks
        self.stage2 = nn.Sequential(
            GatedDWBlock(64, expand_ratio=4, drop_path_prob=0.0),
            GatedDWBlock(64, expand_ratio=4, drop_path_prob=0.0)
        )
        
        # Transition 2: 64 -> 128, 16x16 -> 8x8
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 128 channels, 8x8 spatial
        # 3 GatedDWBlocks
        self.stage3 = nn.Sequential(
            GatedDWBlock(128, expand_ratio=4, drop_path_prob=0.0),
            GatedDWBlock(128, expand_ratio=4, drop_path_prob=0.0),
            GatedDWBlock(128, expand_ratio=4, drop_path_prob=0.0)
        )
        
        # Transition 3: 128 -> 256, 8x8 -> 4x4
        self.trans3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 256 channels, 4x4 spatial
        # 2 GatedDWBlocks
        self.stage4 = nn.Sequential(
            GatedDWBlock(256, expand_ratio=4, drop_path_prob=0.0),
            GatedDWBlock(256, expand_ratio=4, drop_path_prob=0.0)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)
        
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Consistent with Model 1 (Strongest) and Design 1 (Best so far)
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
        
        # Cosine schedule within 1 epoch
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