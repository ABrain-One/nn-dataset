import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)
            
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_se:
            out = self.se(out)
            
        out += identity
        out = self.relu(out)
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

        # CIFAR-style ResNet configuration
        # Widths: 64, 128, 256
        # Depths: 2 blocks per stage (Standard small ResNet)
        widths = [64, 128, 256]
        depths = [2, 2, 2]
        
        # Stem: 3x3 Conv -> Norm -> Activation
        # No max pooling for CIFAR
        self.stem = nn.Sequential(
            nn.Conv2d(channel_number, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True)
        )
        
        self.in_channels = widths[0]
        
        # Stage 1: 32x32 (input size)
        self.stage1 = self._make_layer(widths[0], depths[0], stride=1, dropout=dropout, use_se=False)
        
        # Stage 2: 16x16
        self.stage2 = self._make_layer(widths[1], depths[1], stride=2, dropout=dropout, use_se=False)
        
        # Stage 3: 8x8
        self.stage3 = self._make_layer(widths[2], depths[2], stride=2, dropout=dropout, use_se=True)
        
        # Global Average Pooling and Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[-1], class_number)
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout

    def _make_layer(self, out_channels, depth, stride, dropout, use_se):
        layers = []
        # First block may downsample
        layers.append(BasicBlock(self.in_channels, out_channels, stride, dropout, use_se))
        self.in_channels = out_channels
        for _ in range(depth - 1):
            layers.append(BasicBlock(self.in_channels, out_channels, 1, dropout, use_se))
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