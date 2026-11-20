import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) implementation.
    """
    def __init__(self, drop_prob=0.):
        super().__init__()
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

class ResidualBlock(nn.Module):
    """
    A residual block with a gated depthwise-separable path.
    Uses SiLU activation and BatchNorm.
    """
    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_prob = drop_prob

        # Main path: Depthwise Conv -> Pointwise Conv
        # We use 3x3 depthwise and 1x1 pointwise.
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU(inplace=True)
        
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

        # Shortcut
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.drop_path = DropPath(drop_prob)

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.pw_conv(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.drop_path(out)
        out = out + shortcut
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        n_classes = out_shape[0] if out_shape else 10
        
        # Determine input channels robustly
        # in_shape is typically (C, H, W) or (H, W, C) or (N, C, H, W)
        # For CIFAR-10, C=3.
        if len(in_shape) == 3:
            # If one dim is 3, it's likely channels. 
            # CIFAR is 32x32x3.
            # If in_shape is (3, 32, 32), c_in=3.
            # If in_shape is (32, 32, 3), c_in=3.
            if in_shape[0] == 3:
                c_in = 3
            elif in_shape[-1] == 3:
                c_in = 3
            else:
                c_in = in_shape[0] # Fallback
        elif len(in_shape) == 4:
            c_in = in_shape[1]
        else:
            c_in = 3

        # Architecture Design:
        # We combine the "Fractal" idea (parallel/recursive structure) with "Inverted Bottlenecks" (EfficientNet style) 
        # but implemented in a unique "Gated Residual" manner to avoid copying Design 1/2 or Model 1 directly.
        #
        # Key features:
        # 1. Stem: 3x3 Conv -> BN -> SiLU (No pool, keep 32x32)
        # 2. Stage 1: 2 ResidualBlocks (channels 48 -> 48), Spatial 32x32
        # 3. Transition 1: Stride 2 Conv (48 -> 96), Spatial 16x16
        # 4. Stage 2: 3 ResidualBlocks (channels 96 -> 96), Spatial 16x16
        # 5. Transition 2: Stride 2 Conv (96 -> 192), Spatial 8x8
        # 6. Stage 3: 4 ResidualBlocks (channels 192 -> 192), Spatial 8x8
        # 7. Head: GAP -> FC
        #
        # This channel progression (48-96-192) avoids 64-128-256-512.
        # The blocks are simpler than FractalBlocks (no recursion) but use depthwise-separable convs + gating via residual.
        
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 48 channels
        self.stage1 = nn.Sequential(
            ResidualBlock(48, 48, stride=1, drop_prob=0.0),
            ResidualBlock(48, 48, stride=1, drop_prob=0.0),
        )

        # Transition 1: 48 -> 96, downsample 2x
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )

        # Stage 2: 96 channels
        self.stage2 = nn.Sequential(
            ResidualBlock(96, 96, stride=1, drop_prob=0.0),
            ResidualBlock(96, 96, stride=1, drop_prob=0.0),
            ResidualBlock(96, 96, stride=1, drop_prob=0.0),
        )

        # Transition 2: 96 -> 192, downsample 2x
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )

        # Stage 3: 192 channels
        self.stage3 = nn.Sequential(
            ResidualBlock(192, 192, stride=1, drop_prob=0.0),
            ResidualBlock(192, 192, stride=1, drop_prob=0.0),
            ResidualBlock(192, 192, stride=1, drop_prob=0.0),
            ResidualBlock(192, 192, stride=1, drop_prob=0.0),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, n_classes)

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
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Model 1 (74.66%) used AdamW with lr=0.002, momentum=0.75
        # We use AdamW as per the fixed training setup of the model to improve.
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