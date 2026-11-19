import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        self._init_weights()

        dropout: float = prm['dropout']
        in_channels = in_shape[1]
        spatial = in_shape[2]
        out_classes = out_shape[0]

        # ResNet-style blocks with GroupNorm and SE attention
        # Stages: [32, 16, 8] spatial sizes
        block_counts = [3, 3, 3]
        channels = [64, 128, 256]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 64, 32, block_counts[0], dropout)
        self.layer2 = self._make_layer(64, 128, 16, block_counts[1], dropout)
        self.layer3 = self._make_layer(128, 256, 8, block_counts[2], dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, out_classes),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_ch, out_ch, spatial, num_blocks, dropout):
        blocks = []
        # First block with downsample if needed
        downsample = None
        if in_ch != out_ch or spatial != spatial // 2:
            # Actually spatial is just for reference; we use stride 2 in first block
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, bias=False),
                nn.GroupNorm(32, out_ch),
            )
        blocks.append(BasicBlock(in_ch, out_ch, stride=2, downsample=downsample, dropout=dropout))
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_ch, out_ch, stride=1, downsample=None, dropout=dropout))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32, out_channels)
        self.se = SELayer(out_channels, reduction=4)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.se(out)
        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel, max(int(channel // reduction), 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(int(channel // reduction), 8), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y