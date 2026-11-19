import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.dropout_p = prm['dropout']

        # ResNet-style architecture for CIFAR-10 (32x32)
        # Designed to be strong after 1 epoch and improve with longer training.
        # Uses Wide ResNet style depth for better accuracy.

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Basic Block definition
        # We will use a ResNet-26 style architecture (depth 26, width 1)
        # Or a simpler but effective ResNet-18/20 variant.
        # For CIFAR-10, ResNet-20 or ResNet-32 is standard.
        # Let's go with a robust ResNet-20 (5 blocks of 2 layers each? No, ResNet-20 is 3 stages of 3 blocks? 
        # Standard ResNet-20 for CIFAR: 3 stages, each with 3 basic blocks. Total layers = 2 (stem) + 3*3*2 (blocks) + 1 (fc) = 20.
        # Actually, let's use a slightly deeper network for better 1-epoch convergence and long-term potential.
        # ResNet-32: 3 stages, 5 blocks each.
        # Or Wide ResNet-16-2: 3 stages, 2 blocks each, width 2 (128 channels).
        # Wide ResNets are known to converge faster and have better accuracy.
        # Let's implement a Wide ResNet-16-2 equivalent.
        
        # Stage 1: 128 channels
        # Stage 2: 256 channels
        # Stage 3: 512 channels
        
        # Block structure:
        # BasicBlock:
        #   conv 3x3 -> BN -> ReLU
        #   conv 3x3 -> BN
        #   + shortcut (with 1x1 conv if channels change)
        #   ReLU
        
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout before final layer
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        self.fc = nn.Linear(512, out_shape[0])

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride=stride, dropout_p=self.dropout_p))
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, dropout_p=self.dropout_p))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device).to(torch.float32), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.dropout(out)
        
        out += residual
        out = self.relu(out)
        return out