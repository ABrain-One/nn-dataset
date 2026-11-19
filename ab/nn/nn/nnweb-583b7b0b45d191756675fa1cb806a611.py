import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop_p=0.0):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, 3, stride)
        self.conv2 = ConvBlock(out_ch, out_ch, 3, 1)
        self.drop = nn.Dropout2d(drop_p) if drop_p > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop(out)
        out += identity
        return F.gelu(out)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        
        # Hyperparameters
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.dropout = prm['dropout']

        # Architecture: Deep ResNet-style for CIFAR
        # CIFAR is 32x32. We keep spatial resolution high initially.
        
        # Stem
        # Standard CIFAR ResNet stem is 3x3 conv, stride 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # Stage 1: 64 channels, 32x32
        self.layer1 = self._make_layer(64, 64, blocks=4, stride=1)
        
        # Stage 2: 128 channels, 16x16
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        
        # Stage 3: 256 channels, 8x8
        self.layer3 = self._make_layer(128, 256, blocks=4, stride=2)
        
        # Stage 4: 512 channels, 4x4
        self.layer4 = self._make_layer(256, 512, blocks=4, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, self.dropout))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1, self.dropout))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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