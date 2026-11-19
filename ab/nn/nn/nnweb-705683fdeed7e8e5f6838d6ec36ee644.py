import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.drop(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        channel_number = in_shape[1]
        image_size = in_shape[2]
        class_number = out_shape[0]
        learning_rate = prm['lr']
        momentum = prm['momentum']
        dropout = prm['dropout']

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout
        self.num_classes = class_number

        # Use a smaller initial stride to preserve spatial resolution for small images like CIFAR-10
        # Stem: Conv 3x3, stride 1
        self.stem = nn.Sequential(
            nn.Conv2d(channel_number, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ResNet-18 style architecture adapted for small inputs
        # Stage 1: 64 -> 64 (32x32)
        self.layer1 = self._make_layer(64, 64, 2, stride=1, drop_prob=dropout)
        
        # Stage 2: 64 -> 128 (16x16)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, drop_prob=dropout)
        
        # Stage 3: 128 -> 256 (8x8)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, drop_prob=dropout)
        
        # Stage 4: 256 -> 512 (4x4)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, drop_prob=dropout)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, class_number)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride, drop_prob):
        downsample = None
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, BasicBlock.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(BasicBlock.expansion * out_channels),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample, drop_prob))
        in_channels = BasicBlock.expansion * out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(in_channels, out_channels, drop_prob=drop_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'],)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)

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