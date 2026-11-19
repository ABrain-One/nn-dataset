import math
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = self.fc2(y)
        y = F.sigmoid(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_path_rate=0.0, se_reduction=8):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride, 1)
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, 1, 1)
        self.se = SEBlock(out_channels, se_reduction) if se_reduction > 0 else nn.Identity()

        self.downsample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                ConvBNAct(in_channels, out_channels, 1, stride, 0),
                nn.Identity()  # Keep BN/Act consistent for residual path
            )
        # Note: Standard ResNet uses BN in downsample. We use ConvBNAct which includes Act.
        # To be safe and standard, let's make downsample: Conv1x1 -> BN -> (No Act, as main path has Act)
        # Actually, standard ResNet: Conv -> BN -> (No Act) for downsample.
        # My ConvBNAct has Act. So I need a custom downsample or adjust.
        # Let's just use a simple Conv1x1 + BN for downsample without activation to match standard ResNet residual addition.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        shortcut = self.downsample(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        y = self.drop_path(y)
        y += shortcut
        return y


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        x.div_(keep_prob)
        x.mul_(random_tensor)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride=1, drop_path_rates=None, se_reduction=8):
        super().__init__()
        layers = []
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        else:
            # Ensure length matches
            if len(drop_path_rates) < num_blocks:
                drop_path_rates = drop_path_rates + [drop_path_rates[-1]] * (num_blocks - len(drop_path_rates))
            drop_path_rates = drop_path_rates[:num_blocks]

        for i in range(num_blocks):
            s = stride if i == 0 else 1
            in_ch = in_channels if i == 0 else out_channels
            layers.append(BasicBlock(in_ch, out_channels, s, drop_path_rates[i], se_reduction))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        # Hyperparameters
        dropout_p = prm['dropout']
        momentum = prm['momentum']
        # Note: lr is used in train_setup.
        # We must use 'dropout' and 'momentum' and 'attention_dropout' and 'stochastic_depth_prob'
        # The baseline used 'stochastic_depth_prob' for drop path.
        # The baseline used 'attention_dropout'. Since we are removing attention, we can use it as a general dropout or ignore if not required?
        # Rule: "each of those names ... must still be read from prm and actually used"
        # So I must read 'attention_dropout' and use it. I will use it as an additional dropout on the classifier or feature map.
        attention_dropout = prm['attention_dropout']
        stochastic_depth_prob = prm['stochastic_depth_prob']

        # Architecture: ResNet-18 style with SE and Drop Path
        # CIFAR-10 is 32x32.
        # Stem: Conv 3->64, stride 1 (CIFAR standard, no stride 2 in stem usually, or stride 1)
        # If we use stride 1, we keep 32x32.
        # Block 1: 64->64, 2 blocks
        # Block 2: 64->128, 2 blocks, stride 2
        # Block 3: 128->256, 2 blocks, stride 2
        # Block 4: 256->512, 2 blocks, stride 2
        # Final size: 32 -> 16 -> 8 -> 4.
        
        # Let's stick to a solid ResNet-18 configuration which is robust and fast.
        # Channels: [64, 128, 256, 512]
        # Blocks: [2, 2, 2, 2]
        
        # Drop path rates: linearly increasing from 0 to stochastic_depth_prob
        total_blocks = 8
        drop_path_rates = [stochastic_depth_prob * i / total_blocks for i in range(total_blocks)]
        
        # Assign rates to blocks
        # Block 1: rates[0], rates[1]
        # Block 2: rates[2], rates[3]
        # Block 3: rates[4], rates[5]
        # Block 4: rates[6], rates[7]

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 64, 3, 1, 1),
        )

        self.layer1 = ResNetBlock(64, 64, 2, stride=1, drop_path_rates=drop_path_rates[0:2], se_reduction=8)
        self.layer2 = ResNetBlock(64, 128, 2, stride=2, drop_path_rates=drop_path_rates[2:4], se_reduction=8)
        self.layer3 = ResNetBlock(128, 256, 2, stride=2, drop_path_rates=drop_path_rates[4:6], se_reduction=8)
        self.layer4 = ResNetBlock(256, 512, 2, stride=2, drop_path_rates=drop_path_rates[6:8], se_reduction=8)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Use attention_dropout here to satisfy the requirement
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(attention_dropout),
            nn.Linear(512, num_classes),
        )
        
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}