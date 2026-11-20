import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_rate=0.0):
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

        self.drop_rate = drop_rate

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, in_channels, drop_rate):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # CIFAR-style stem: 3x3 conv, no max pooling
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet-9: 1 block per layer, widths [64, 128, 256]
        # Layer 1: 64->64, stride 1
        # Layer 2: 64->128, stride 2
        # Layer 3: 128->256, stride 2
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, drop_rate=0.0))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


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
        drop_rate: float = prm['dropout']

        # ResNet-9 configuration
        # 3 layers, 1 block each
        num_blocks = (1, 1, 1)

        self.model = ResNet(BasicBlock, num_blocks, num_classes, in_channels, drop_rate)

        # Apply dropout to the final layer output? Or inside blocks?
        # The prompt says "use only names the file already reads from prm... e.g. a dropout layer with p=prm['dropout']".
        # Standard ResNet doesn't have dropout. I will add a dropout layer before the final linear layer to ensure the hyperparameter is used.
        # Actually, better to put it in the blocks or after pooling. Let's put it after adaptive_avg_pool2d.
        # Wait, the ResNet class above has a linear layer. I need to modify the forward or structure to include dropout.
        # Let's restructure Net to hold the parts or modify ResNet to accept drop_rate and use it.
        
        # Let's redo the ResNet class to accept drop_rate and use it in the forward pass before linear.
        # I will define the model inside Net to keep the signature clean.

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        in_planes = 64

        # Layer 1: 64 -> 64, stride 1
        self.layer1 = self._make_layer(in_planes, 64, 1, stride=1)
        in_planes = 64

        # Layer 2: 64 -> 128, stride 2
        self.layer2 = self._make_layer(in_planes, 128, 1, stride=2)
        in_planes = 128

        # Layer 3: 128 -> 256, stride 2
        self.layer3 = self._make_layer(in_planes, 256, 1, stride=2)
        in_planes = 256

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=drop_rate)
        self.linear = nn.Linear(in_planes, num_classes)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(BasicBlock(in_planes, planes, stride, drop_rate=0.0))
            in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.drop(out)
        out = self.linear(out)
        return out