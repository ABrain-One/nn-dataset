import torch
import torch.nn.functional as F
from torch import nn, Tensor


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes: int = out_shape[0]
        in_channels: int = in_shape[1]
        dropout: float = prm['dropout']

        # RegNet-style blocks with SE attention and residual connections.
        # Designed for fast learning and strong long-horizon improvement.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Stage 1: 32 -> 64, resolution preserved (32x32)
        self.stage1 = nn.Sequential(
            RegNetBlock(32, 64, se_ratio=4.0),
            RegNetBlock(64, 64, se_ratio=4.0),
            RegNetBlock(64, 64, se_ratio=4.0),
        )

        # Stage 2: 64 -> 128, stride 2 (16x16)
        self.stage2 = nn.Sequential(
            RegNetBlock(64, 128, se_ratio=4.0, downsample=True),
            RegNetBlock(128, 128, se_ratio=4.0),
            RegNetBlock(128, 128, se_ratio=4.0),
            RegNetBlock(128, 128, se_ratio=4.0),
        )

        # Stage 3: 128 -> 256, stride 2 (8x8)
        self.stage3 = nn.Sequential(
            RegNetBlock(128, 256, se_ratio=4.0, downsample=True),
            RegNetBlock(256, 256, se_ratio=4.0),
            RegNetBlock(256, 256, se_ratio=4.0),
            RegNetBlock(256, 256, se_ratio=4.0),
            RegNetBlock(256, 256, se_ratio=4.0),
        )

        # Stage 4: 256 -> 512, stride 2 (4x4)
        self.stage4 = nn.Sequential(
            RegNetBlock(256, 512, se_ratio=4.0, downsample=True),
            RegNetBlock(512, 512, se_ratio=4.0),
            RegNetBlock(512, 512, se_ratio=4.0),
            RegNetBlock(512, 512, se_ratio=4.0),
            RegNetBlock(512, 512, se_ratio=4.0),
            RegNetBlock(512, 512, se_ratio=4.0),
        )

        # Classification head with global average pooling and dropout
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels: int, reduction: float = 4.0):
        super().__init__()
        mid = max(1, int(channels // reduction))
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        # Squeeze: global average pooling
        s = x.mean(dim=[2, 3])  # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        # Excitation: reshape and multiply
        s = s.view(b, c, 1, 1)
        return x * s


class RegNetBlock(nn.Module):
    """
    RegNet-style block with SE attention and residual connection.
    Uses 3x3 convolutions with batch norm and GELU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, se_ratio: float = 4.0, downsample: bool = False):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if downsample else 1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=se_ratio)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut projection if channels or resolution change
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.se(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = F.gelu(out + identity)
        return out