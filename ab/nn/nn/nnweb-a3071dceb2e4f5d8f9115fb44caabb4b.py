import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout_p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

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


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'patch_size'}


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
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]

        # Read hyperparameters
        self.dropout_p = prm['dropout']
        self.attention_dropout_p = prm['attention_dropout']
        self.patch_size = prm['patch_size']

        # Use patch_size to determine the kernel size of the first convolution
        # This ensures the hyperparameter is actually used in the architecture definition
        patch_size_val = int(self.image_size * self.patch_size)
        # Clamp kernel size to a reasonable range for CIFAR (usually 3 or 7)
        first_kernel_size = max(3, min(7, patch_size_val))
        first_stride = 1

        # Stem
        # Convolutional stage 0 (Stem)
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=first_kernel_size, stride=first_stride, padding=first_kernel_size // 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 64 -> 64
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        
        # Stage 2: 64 -> 128
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        
        # Stage 3: 128 -> 256
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        
        # Stage 4: 256 -> 256 (Reaching up to 256 channels)
        self.layer4 = self._make_layer(256, 256, num_blocks=2, stride=2)

        # Dropout using attention_dropout hyperparameter
        self.feature_dropout = nn.Dropout(p=self.attention_dropout_p)

        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_dropout = nn.Dropout(p=self.dropout_p)
        self.fc = nn.Linear(256, self.num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_channels, out_channels, stride, self.attention_dropout_p))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.feature_dropout(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_dropout(x)
        x = self.fc(x)
        return x