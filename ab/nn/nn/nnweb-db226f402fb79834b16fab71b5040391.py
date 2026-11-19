import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    """Lightweight Squeeze-and-Excitation block"""
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
    Recursive depthwise convolution block inspired by Model 1.
    Base case: DW Conv -> BN -> SiLU -> 1x1 Conv -> BN
    Recursive case: Parallel branches processed by smaller blocks,
    combined via simple averaging, then passed through core.
    """
    def __init__(self, depth, channels, use_se=False):
        super().__init__()
        self.depth = depth
        self.channels = channels

        # Core processing
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # Optional SE block
        self.se = SEBlock(channels, reduction=8) if use_se else nn.Identity()

        if depth > 1:
            self.left = FractalDWBlock(depth - 1, channels, use_se=False)
            self.right = FractalDWBlock(depth - 1, channels, use_se=False)

    def forward(self, x):
        if self.depth > 1:
            left_out = self.left(x)
            right_out = self.right(x)
            # Simple averaging (like Model 1's FractalDropPath with prob=0)
            x = (left_out + right_out) / 2.0

        # Core
        y = self.dw_conv(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.pw_conv(y)
        y = self.bn2(y)
        y = self.se(y)

        # Residual connection
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')

        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10

        # Stem: 3 -> 48 with 5x5 kernel for early context (inspired by Model 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 48 channels, 32x32 -> 32x32
        # Fractal depth 2
        self.stage1 = FractalDWBlock(depth=2, channels=48, use_se=False)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Transition 1: 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )

        # Stage 2: 96 channels, 16x16 -> 16x16
        # Fractal depth 3
        self.stage2 = FractalDWBlock(depth=3, channels=96, use_se=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Transition 2: 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )

        # Stage 3: 192 channels, 8x8 -> 8x8
        # Fractal depth 3
        self.stage3 = FractalDWBlock(depth=3, channels=192, use_se=True)

        # Head: GAP -> small FC -> classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 96),
            nn.SiLU(inplace=True)
        )
        self.fc2 = nn.Linear(96, n_classes)

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
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # AdamW with momentum mapped to beta1 (like Model 1)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )

        # Estimate total steps for CIFAR-10 (50,000 images)
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size

        # Cosine Annealing schedule
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

        # Create scheduler if not already created
        if not hasattr(self, 'scheduler'):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=prm_lr * 0.01 if 'prm_lr' in dir() else 0.00002
            )

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping (like Model 1)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)

            self.optimizer.step()
            self.scheduler.step()