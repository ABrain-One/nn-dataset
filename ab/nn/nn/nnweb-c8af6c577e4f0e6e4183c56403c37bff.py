import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
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
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.class_number = out_shape[0]
        in_channels = in_shape[1]

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 4 Convolutional Stages
        # Stage 1: 32x32, 64 channels
        self.stage1 = nn.Sequential(
            ResBlock(64, 64, stride=1)
        )

        # Stage 2: 16x16, 128 channels
        self.stage2 = nn.Sequential(
            ResBlock(64, 128, stride=2)
        )

        # Stage 3: 8x8, 256 channels
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, stride=2)
        )

        # Stage 4: 4x4, 512 channels (Standard CNN depth often goes higher, but description says "up to 256". 
        # However, "4 convolution stages" usually implies downsampling. 
        # Let's stick to the description "up to 256 channels". 
        # If I use 256 in stage 3, stage 4 could be 256 or higher. 
        # Common CIFAR10 CNNs (like the one referenced) often have:
        # Conv(64) -> Conv(128) -> Conv(256) -> Conv(512) is typical for "deep".
        # But the description explicitly says "up to 256 channels".
        # Let's interpret "4 convolution stages" as 4 blocks of convolutions.
        # If max is 256, maybe the stages are 64, 128, 256, 256? Or 64, 128, 256, 512?
        # The source link (11Cpss) typically uses:
        # Conv(64) -> Pool -> Conv(128) -> Pool -> Conv(256) -> Pool -> Conv(512) -> FC
        # Wait, the description says "up to 256". Let's look at the specific repo if possible.
        # Repo: https://github.com/11Cpss/cifar10-from-scratch-cnn
        # That repo usually has a simple CNN:
        # Conv(64) -> BN -> ReLU -> Pool
        # Conv(128) -> BN -> ReLU -> Pool
        # Conv(256) -> BN -> ReLU -> Pool
        # Conv(512) -> BN -> ReLU -> Pool
        # This contradicts "up to 256".
        # Let's re-read carefully: "4 convolution stages (up to 256 channels)".
        # Maybe it means the *maximum* width is 256.
        # Let's assume the stages are: 64, 128, 256, 256.
        # Or perhaps the "4 stages" refers to the 4 conv layers in a standard simple CNN?
        # A "standard choice" for a CNN with 4 stages and max 256 channels on CIFAR-10:
        # Stage 1: 64 channels
        # Stage 2: 128 channels
        # Stage 3: 256 channels
        # Stage 4: 256 channels (keeping width constant or increasing slightly, but capped at 256)
        
        # Let's implement:
        # Stage 1: 64
        # Stage 2: 128
        # Stage 3: 256
        # Stage 4: 256
        
        # Downsampling: Standard CIFAR-10 CNNs downsample at the start of each stage (except the first).
        # Input 32x32
        # Stage 1: 32x32, 64 ch
        # Stage 2: 16x16, 128 ch
        # Stage 3: 8x8, 256 ch
        # Stage 4: 4x4, 256 ch
        
        self.stage4 = nn.Sequential(
            ResBlock(256, 256, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, self.class_number)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

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