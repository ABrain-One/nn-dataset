import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}


class SEBlock(nn.Module):
    """Squeeze-and-Excitation with SiLU for lightweight channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FractalDWBlock(nn.Module):
    """
    Recursive depthwise-separable block with SE attention.
    Base (depth=1): DW Conv(3x3) -> BN -> SiLU -> PW Conv(1x1) -> BN -> SE
    Recursive (depth>1): Two parallel sub-branches (left, right) each processed
    by a smaller FractalDWBlock, then combined via learned gating, then core.
    """
    def __init__(self, depth, channels, use_se=True):
        super().__init__()
        self.depth = depth
        self.channels = channels

        # Core: depthwise 3x3 -> BN -> SiLU -> pointwise 1x1 -> BN
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                 groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE attention on the output
        self.se = SEBlock(channels, reduction=8) if use_se else nn.Identity()

        # Recursive branches
        if depth > 1:
            self.left = FractalDWBlock(depth - 1, channels, use_se=False)
            self.right = FractalDWBlock(depth - 1, channels, use_se=False)
            # Learned gating for combining parallel branches
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        # Recursive combination
        if self.depth > 1:
            left_out = self.left(x)
            right_out = self.right(x)
            gate = self.gate(x).unsqueeze(-1).unsqueeze(-1)
            x = gate * left_out + (1.0 - gate) * right_out

        # Core processing
        y = self.dw_conv(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.pw_conv(y)
        y = self.bn2(y)
        y = self.se(y)

        # Residual connection
        return x + y


class MultiScaleFusion(nn.Module):
    """
    Fuses features from multiple scales using parallel 1x1 convolutions
    with different receptive fields, then combines via learned weights.
    """
    def __init__(self, channels):
        super().__init__()
        # Parallel branches with different kernel sizes
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        # Learned fusion weights
        self.fuse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 2, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        # Compute fusion weights
        w = self.fuse(x).view(-1, 2, 1, 1)
        out = w[:, 0:1] * b1 + w[:, 1:2] * b2
        return out


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')

        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10

        # Stem: 3 -> 32 with larger kernel for early context
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 32 channels, 32x32
        # Fractal depth 2 for multi-scale extraction
        self.stage1 = nn.Sequential(
            FractalDWBlock(depth=2, channels=32, use_se=True),
            FractalDWBlock(depth=2, channels=32, use_se=True)
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Transition 1: 32 -> 64
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        # Stage 2: 64 channels, 16x16
        # Fractal depth 2 for richer multi-scale features
        self.stage2 = nn.Sequential(
            FractalDWBlock(depth=2, channels=64, use_se=True),
            FractalDWBlock(depth=2, channels=64, use_se=True),
            FractalDWBlock(depth=2, channels=64, use_se=True)
        )
        # Multi-scale fusion after stage 2
        self.fuse1 = MultiScaleFusion(64)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Transition 2: 64 -> 128
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )

        # Stage 3: 128 channels, 8x8
        # Fractal depth 2
        self.stage3 = nn.Sequential(
            FractalDWBlock(depth=2, channels=128, use_se=True),
            FractalDWBlock(depth=2, channels=128, use_se=True),
            FractalDWBlock(depth=2, channels=128, use_se=True)
        )
        self.fuse2 = MultiScaleFusion(128)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Transition 3: 128 -> 256
        self.trans3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )

        # Stage 4: 256 channels, 4x4
        # Fractal depth 1 (lighter at the end)
        self.stage4 = nn.Sequential(
            FractalDWBlock(depth=1, channels=256, use_se=True),
            FractalDWBlock(depth=1, channels=256, use_se=True)
        )
        self.fuse3 = MultiScaleFusion(256)

        # Head: GAP + classifier
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
        # Apply normalization if transform indicates it
        # CIFAR-10 mean/std
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470

        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.fuse1(x)
        x = self.pool2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.fuse2(x)
        x = self.pool3(x)
        x = self.trans3(x)
        x = self.stage4(x)
        x = self.fuse3(x)
        x = self.gap(x)
        x = self.classifier(x)
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

        # Estimate total steps for CIFAR-10 (50,000 images)
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size

        # Cosine annealing within the single epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=prm['lr'] * 0.01
        )

        return self.optimizer

    def learn(self, train_data):
        self.train()

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)

            self.optimizer.step()
            self.scheduler.step()