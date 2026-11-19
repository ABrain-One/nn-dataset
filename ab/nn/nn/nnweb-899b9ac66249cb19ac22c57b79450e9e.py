import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        return out


class CIFARResidualNet(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(CIFARResidualNet, self).__init__()
        
        # Stem: 3x3 Conv2d, norm, activation
        # CIFAR input is 32x32. To get to 32x32 (Stage 1), we use stride 1.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: residual blocks at 32x32
        # We choose 2 blocks per stage as a standard small depth (like ResNet-14 or similar small variants).
        # Widths: 64, 128, 256 is a common progression.
        self.stage1 = self._make_stage(64, 64, num_blocks=2, stride=1)
        
        # Stage 2: residual blocks at 16x16
        # Downsample from 32 to 16 using stride 2 in the first block of the stage.
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2)
        
        # Stage 3: residual blocks at 8x8
        # Downsample from 16 to 8 using stride 2 in the first block of the stage.
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block may need to downsample and change channels
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        in_channels = out_channels
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        channel_number = in_shape[1]
        class_number = out_shape[0]
        
        self.model = CIFARResidualNet(
            num_classes=class_number, 
            in_channels=channel_number
        )
        
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