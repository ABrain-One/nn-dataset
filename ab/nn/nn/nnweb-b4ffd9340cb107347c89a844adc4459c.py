import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        random_tensor.div_(keep_prob)
        return x * random_tensor

class HybridBlock(nn.Module):
    """
    Combines the efficiency of depthwise-separable convolutions (from Model 1/3)
    with the robustness of residual connections (from Design 1/2).
    Uses a 1x1 expansion and 3x3 depthwise conv, followed by 1x1 projection.
    """
    def __init__(self, channels, expand_ratio=2, drop_prob=0.0):
        super(HybridBlock, self).__init__()
        self.drop_prob = drop_prob
        
        # Inverted Bottleneck structure
        hidden_dim = int(channels * expand_ratio)
        
        # Expansion
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise Conv
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # Projection
        self.conv3 = nn.Conv2d(hidden_dim, channels, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.act = nn.SiLU(inplace=True)
        self.drop_path = DropPath(drop_prob)

    def forward(self, x):
        identity = x
        
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        
        y = self.conv3(y)
        y = self.bn3(y)
        
        y = self.drop_path(y)
        
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        
        # Derive input channels robustly
        c_in = 3
        if len(in_shape) == 3:
            if in_shape[0] == 3: c_in = 3
            elif in_shape[-1] == 3: c_in = 3
            else: c_in = in_shape[1]
        elif len(in_shape) == 4:
            c_in = in_shape[1]
            
        n_classes = out_shape[0] if out_shape else 10

        # Architecture:
        # Stem: 3 -> 32 (smaller start to allow more depth)
        # Stage 1: 32 ch, 32x32 -> 16x16 (via stride 2 in first block? No, keep spatial, pool later)
        # Let's use a standard progression but with HybridBlocks.
        # To avoid 64-128-256-512, we use 32-64-128-256.
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 channels, 32x32
        # 2 blocks
        self.stage1 = nn.Sequential(
            HybridBlock(32, expand_ratio=2, drop_prob=0.0),
            HybridBlock(32, expand_ratio=2, drop_prob=0.0)
        )
        
        # Transition 1: 32 -> 64, 32x32 -> 16x16
        # Use stride 2 in the first block of the next stage? Or a separate pooling layer?
        # Model 1 used AvgPool. Design 1 used AvgPool in transition.
        # Let's use a strided 1x1 conv for transition to preserve spatial info better than pooling?
        # Or just AvgPool to match Model 1's success.
        self.trans1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 64 channels, 16x16
        # 3 blocks
        self.stage2 = nn.Sequential(
            HybridBlock(64, expand_ratio=2, drop_prob=0.0),
            HybridBlock(64, expand_ratio=2, drop_prob=0.0),
            HybridBlock(64, expand_ratio=2, drop_prob=0.0)
        )
        
        # Transition 2: 64 -> 128, 16x16 -> 8x8
        self.trans2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 128 channels, 8x8
        # 4 blocks
        self.stage3 = nn.Sequential(
            HybridBlock(128, expand_ratio=2, drop_prob=0.0),
            HybridBlock(128, expand_ratio=2, drop_prob=0.0),
            HybridBlock(128, expand_ratio=2, drop_prob=0.0),
            HybridBlock(128, expand_ratio=2, drop_prob=0.0)
        )
        
        # Transition 3: 128 -> 256, 8x8 -> 4x4
        self.trans3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # Stage 4: 256 channels, 4x4
        # 2 blocks
        self.stage4 = nn.Sequential(
            HybridBlock(256, expand_ratio=2, drop_prob=0.0),
            HybridBlock(256, expand_ratio=2, drop_prob=0.0)
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
        
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            
            self.optimizer.step()
            scheduler.step()