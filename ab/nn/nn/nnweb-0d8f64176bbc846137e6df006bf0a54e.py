import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from typing import Callable, List, Optional, Sequence, Tuple


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_prob=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.drop_prob = drop_prob

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.drop_prob > 0 and self.training:
            # Stochastic depth: drop the entire residual path
            keep_prob = 1.0 - self.drop_prob
            shape = (out.shape[0],) + (1,) * (out.ndim - 1)
            mask = torch.bernoulli(torch.full(shape, keep_prob, device=out.device))
            out = out * mask / keep_prob if keep_prob > 0 else out
        
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3, 
                 dropout=0.0, attention_dropout=0.0, stochastic_depth_prob=0.0):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # CIFAR-10 specific stem: 3x3 conv, no maxpool
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, drop_prob=stochastic_depth_prob * 0.1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, drop_prob=stochastic_depth_prob * 0.2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, drop_prob=stochastic_depth_prob * 0.3)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, drop_prob=stochastic_depth_prob * 0.4)

        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        # Hyperparameters
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        
        # Simple SE-like attention block to use attention_dropout
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=attention_dropout),
            nn.Linear(128, 512),
            nn.Sigmoid(),
            nn.Flatten(start_dim=1, end_dim=1), # Keep spatial dims? No, SE usually returns channel weights
            # Let's fix SE to return (B, C, 1, 1)
        )
        # Correct SE implementation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=attention_dropout),
            nn.Linear(128, 512),
            nn.Sigmoid(),
            nn.Unflatten(1, (512, 1, 1))
        )

    def _make_layer(self, block, planes, num_blocks, stride, drop_prob):
        layers = []
        layers.append(block(self.in_planes, planes, stride, drop_prob))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, 1, drop_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Apply SE attention
        se_out = self.se(out)
        out = out * se_out
        
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.linear(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        
        # Read hyperparameters
        self.lr = prm['lr']
        self.momentum = prm['momentum']
        self.dropout = prm['dropout']
        self.attention_dropout = prm['attention_dropout']
        self.stochastic_depth_prob = prm['stochastic_depth_prob']
        
        # Build ResNet-18 (2, 2, 2, 2)
        self.model = ResNet(
            BasicBlock, 
            [2, 2, 2, 2], 
            num_classes=self.num_classes, 
            in_channels=self.in_channels,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            stochastic_depth_prob=self.stochastic_depth_prob
        )
        
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

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


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}