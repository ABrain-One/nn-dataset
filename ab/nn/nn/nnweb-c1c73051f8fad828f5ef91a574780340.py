import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class ChannelAttention(nn.Module):
    """SE-style channel attention with SiLU."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        avg = x.mean(dim=(2, 3), keepdim=True)
        y = self.fc2(self.act(self.fc1(avg)))
        return x * torch.sigmoid(y)

class SpatialAttention(nn.Module):
    """Simple spatial attention (avg+max -> 7x7 conv -> sigmoid)."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        cat = torch.cat([avg, mx], dim=1)
        return torch.sigmoid(self.conv(cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(k)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x) * x
        return x

class InvertedBlock(nn.Module):
    """
    EfficientNet-style inverted residual:
    1x1 expand -> BN -> SiLU -> 3x3 depthwise -> BN -> SiLU -> [CBAM] -> 1x1 project -> BN
    with residual if shapes match.
    """
    def __init__(self, in_ch, out_ch, stride, expand_ratio=4, use_cbam=False):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden = int(round(in_ch * expand_ratio))

        self.expand = nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)

        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)

        self.cbam = CBAM(hidden, reduction=16) if use_cbam else nn.Identity()

        self.project = nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        inp = x
        x = self.act(self.bn1(self.expand(x)))
        x = self.act(self.bn2(self.dw(x)))
        x = self.cbam(x)
        x = self.bn3(self.project(x))
        if self.use_residual:
            x = x + inp
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        self.batch_size = prm.get('batch', 128)

        c_in = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10

        # Stem: 3 -> 32, keep resolution (no stride) to preserve small-image detail
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 32 channels, 2 blocks (no stride)
        self.stage1 = nn.Sequential(
            InvertedBlock(32, 32, stride=1, expand_ratio=4, use_cbam=False),
            InvertedBlock(32, 32, stride=1, expand_ratio=4, use_cbam=True)
        )

        # Downsample 2x: 32 -> 64
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(2, stride=2)
        )

        # Stage 2: 64 channels, 3 blocks
        self.stage2 = nn.Sequential(
            InvertedBlock(64, 64, stride=1, expand_ratio=4, use_cbam=False),
            InvertedBlock(64, 64, stride=1, expand_ratio=4, use_cbam=True),
            InvertedBlock(64, 64, stride=1, expand_ratio=4, use_cbam=True)
        )

        # Downsample 2x: 64 -> 128
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(2, stride=2)
        )

        # Stage 3: 128 channels, 4 blocks
        self.stage3 = nn.Sequential(
            InvertedBlock(128, 128, stride=1, expand_ratio=4, use_cbam=False),
            InvertedBlock(128, 128, stride=1, expand_ratio=4, use_cbam=True),
            InvertedBlock(128, 128, stride=1, expand_ratio=4, use_cbam=True),
            InvertedBlock(128, 128, stride=1, expand_ratio=4, use_cbam=True)
        )

        # Downsample 2x: 128 -> 256
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(2, stride=2)
        )

        # Stage 4: 256 channels, 2 blocks
        self.stage4 = nn.Sequential(
            InvertedBlock(256, 256, stride=1, expand_ratio=4, use_cbam=True),
            InvertedBlock(256, 256, stride=1, expand_ratio=4, use_cbam=True)
        )

        # Head: GAP + linear
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)

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
        # Apply normalization if the transform indicates it (CIFAR-10 mean/std)
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470

        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # AdamW with momentum mapped to beta1 (Model 1's recipe)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )

        # Estimate total steps for one epoch (CIFAR-10 has 50k train images)
        batch_size = prm.get('batch', 128)
        total_steps = max(1, 50000 // batch_size)

        # Cosine annealing within the single epoch, with a small floor
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=prm['lr'] * 0.01
        )

        return self.optimizer

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()