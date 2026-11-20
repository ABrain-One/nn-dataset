import torch
from torch import nn, Tensor
from typing import List


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
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
        in_channels = in_shape[1]

        # Use hyperparameters
        dropout = prm['dropout']
        norm_eps = prm['norm_eps']
        norm_momentum = prm['norm_momentum']
        # lr and momentum are used in train_setup

        # Define BatchNorm with specific eps and momentum
        self.bn_init = nn.BatchNorm2d(in_channels, eps=norm_eps, momentum=norm_momentum)
        
        # Stage 1: 3x3 conv, 64 channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=norm_eps, momentum=norm_momentum)
        self.relu = nn.ReLU(inplace=True)

        # Stage 2: Residual Block 64 -> 128, stride 2
        self.res_block1 = ResidualBlock(64, 128, stride=2)
        # Update BN params for residual blocks? The class uses default BN.
        # We need to ensure the BN layers inside ResidualBlock use the correct eps/momentum.
        # Since ResidualBlock creates its own BN layers, we should re-initialize them or pass params.
        # Let's modify ResidualBlock to accept norm params or just iterate and update.
        # Simpler: iterate through modules and update BN params.
        
        # Stage 3: Residual Block 128 -> 256, stride 2
        self.res_block2 = ResidualBlock(128, 256, stride=2)

        # Stage 4: 3x3 conv, 256 channels (No downsampling mentioned for final stage usually, or just feature extraction)
        # Description says "4 convolution stages". 
        # Stage 1: Conv
        # Stage 2: Res (downsample)
        # Stage 3: Res (downsample)
        # Stage 4: Conv (same size)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256, eps=norm_eps, momentum=norm_momentum)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)

        # Update all BatchNorm layers to use the specific eps and momentum from prm
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = norm_eps
                m.momentum = norm_momentum

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn_init(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'norm_eps', 'norm_momentum'}