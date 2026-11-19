import math
import torch
from torch import nn, Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, BasicBlock.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(BasicBlock.expansion * out_channels)
            )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


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
        num_classes = out_shape[0]
        dropout = prm['dropout']

        # Stem: Conv 3x3, 64 channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 64 channels
        self.stage1 = nn.Sequential(
            BasicBlock(64, 64, stride=1, dropout=dropout),
            BasicBlock(64, 64, stride=1, dropout=dropout)
        )

        # Stage 2: 128 channels
        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, dropout=dropout),
            BasicBlock(128, 128, stride=1, dropout=dropout)
        )

        # Stage 3: 256 channels
        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, dropout=dropout),
            BasicBlock(256, 256, stride=1, dropout=dropout)
        )

        # Stage 4: 256 channels
        self.stage4 = nn.Sequential(
            BasicBlock(256, 256, stride=1, dropout=dropout),
            BasicBlock(256, 256, stride=1, dropout=dropout)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

        # Weight initialization
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}