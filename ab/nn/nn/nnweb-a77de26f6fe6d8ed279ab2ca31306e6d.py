import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class BasicBlock(nn.Module):
    """Standard ResNet basic block with 2x2 convolutions and 1x1 shortcut."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet9(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(ResNet9, self).__init__()
        self.in_planes = 64

        # Initial convolution: 3x3, stride 1 (standard for CIFAR ResNets)
        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # ResNet-9 structure: 4 layers of 2 blocks each (total 8 blocks + initial conv = depth 9)
        # Layer 1: 64 channels, no downsampling
        self.layer1 = self._make_layer(64, 2, stride=1)
        # Layer 2: 128 channels, downsample
        self.layer2 = self._make_layer(128, 2, stride=2)
        # Layer 3: 256 channels, downsample
        self.layer3 = self._make_layer(256, 2, stride=2)
        # Layer 4: 512 channels, no downsampling
        self.layer4 = self._make_layer(512, 2, stride=1)

        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
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


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        channel_number = in_shape[1]
        class_number = out_shape[0]
        self.model = ResNet9(num_classes=class_number, in_channels=channel_number)
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
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