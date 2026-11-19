import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.dropout = 0.0  # Standard choice for ResNet-9; dropout is not typically used in the core architecture

        # ResNet-9 configuration
        # Total layers: 1 (stem) + 8 (residual blocks) + 1 (fc) = 9
        # Block distribution: 2 blocks per stage, 4 stages
        # Widths: [64, 128, 256, 512] is standard for CIFAR ResNets.
        # The description mentions "configurable width (e.g., 0.75 scale)", but without a specific width hyperparameter in `prm`,
        # we use the standard base widths for a compact ResNet.
        widths = [64, 128, 256, 512]
        depths = [2, 2, 2, 2]  # 2 blocks per stage, 4 stages = 8 residual blocks

        self.in_planes = 64

        # Stem: 3x3 conv, BN, ReLU. No downsampling (stride 1) to preserve spatial info as per description.
        self.conv1 = nn.Conv2d(self.in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Residual stages
        self.layer1 = self._make_layer(widths[0], depths[0], stride=1)
        self.layer2 = self._make_layer(widths[1], depths[1], stride=2)
        self.layer3 = self._make_layer(widths[2], depths[2], stride=2)
        self.layer4 = self._make_layer(widths[3], depths[3], stride=2)

        self.linear = nn.Linear(widths[3], self.num_classes)

    def _make_layer(self, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, out_planes, stride, self.dropout))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()