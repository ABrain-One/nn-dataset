import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


def supported_hyperparameters():
    return {'lr', 'momentum'}


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dropout=0.0, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        # Hyperparameter usage: We use momentum in train_setup. 
        # We also introduce a small dropout to regularize the fast learning process, 
        # though strictly speaking, the prompt asks to use names in supported_hyperparameters.
        # The baseline only had 'lr' and 'momentum'. I will stick to those two to be safe, 
        # or I can add 'dropout' to supported_hyperparameters? 
        # The prompt says: "keep supported_hyperparameters ... exactly as they are".
        # So I CANNOT add 'dropout' to supported_hyperparameters.
        # I MUST use only 'lr' and 'momentum' from prm.
        # Therefore, I will NOT use a dropout layer unless I hardcode it, but hardcoding is fine as long as I don't read a new name.
        # Wait, the prompt says: "each of those names ... must still be read from prm and actually used".
        # It does NOT say I cannot use other constants. But to be safe and clean, I will just use lr and momentum.
        # I will design a ResNet-18 style architecture.

        in_channels = in_shape[1]
        num_classes = out_shape[0]

        # ResNet-18 Architecture
        # CIFAR-10 is 32x32.
        # Standard ResNet for CIFAR uses 3x3 convs with stride 1 in the first block, then stride 2.
        # Input: 32x32
        # Block 1: 32x32 -> 32x32 (stride 1)
        # Block 2: 32x32 -> 16x16 (stride 2)
        # Block 3: 16x16 -> 8x8 (stride 2)
        # Block 4: 8x8 -> 4x4 (stride 2)
        
        # Let's stick to a standard ResNet-18 config adapted for CIFAR.
        # Channels: 64, 128, 256, 512
        
        self.in_planes = 64

        # Initial conv: 3x3, stride 1, padding 1. No maxpool.
        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Residual Blocks
        # Layer 1: 64 channels, 32x32
        self.layer1 = self._make_layer(64, 2, stride=1)
        # Layer 2: 128 channels, 16x16
        self.layer2 = self._make_layer(128, 2, stride=2)
        # Layer 3: 256 channels, 8x8
        self.layer3 = self._make_layer(256, 2, stride=2)
        # Layer 4: 512 channels, 4x4
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.linear = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, plane, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, plane, stride))
            self.in_planes = plane
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

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