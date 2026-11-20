import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection for dimension matching if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
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
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        
        # Hyperparameters
        self.dropout_rate = prm['dropout']
        self.attention_dropout = prm['attention_dropout'] # Read to satisfy LEMUR, though not used in CNN
        self.patch_size = prm['patch_size'] # Read to satisfy LEMUR, though not used in CNN

        # Architecture: 4 Conv Stages
        # Stage 1: 3 -> 64, stride 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64, stride=1)
        )

        # Stage 2: 64 -> 128, stride 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128, stride=1)
        )

        # Stage 3: 128 -> 256, stride 2
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256, stride=1)
        )

        # Stage 4: 256 -> 256, stride 2
        # Input 32x32 -> 32 -> 16 -> 8 -> 4
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256, stride=1)
        )

        # Head
        # Calculate feature map size dynamically based on input size
        # 32 -> 32 -> 16 -> 8 -> 4
        # If input is not 32, we derive the final spatial size.
        # However, CIFAR-10 is 32x32. The prompt says "derive what you need from in_shape".
        # We will use AdaptiveAvgPool2d to be safe for any input size, or calculate exact size.
        # Standard CIFAR-10 CNNs often use a fixed head size based on 32x32 input (4x4).
        # To be robust, we use AdaptiveAvgPool2d(1) or calculate the size.
        # Let's calculate the size to be precise for the Linear layer.
        
        # Simulate the spatial reduction:
        # Stage 1: stride 1 -> H/1
        # Stage 2: stride 2 -> H/2
        # Stage 3: stride 2 -> H/4
        # Stage 4: stride 2 -> H/8
        
        # Wait, the description says "4 convolution stages". 
        # Typical CIFAR-10 ResNet-like structures:
        # 1. Conv 3x3 (S1)
        # 2. Conv 3x3 (S2)
        # 3. Conv 3x3 (S3)
        # 4. Conv 3x3 (S4)
        
        # If we follow the "stride 2 in deeper stages" pattern:
        # Input: 32
        # Stage 1: 32 (stride 1)
        # Stage 2: 16 (stride 2)
        # Stage 3: 8 (stride 2)
        # Stage 4: 4 (stride 2)
        
        # Final feature map is 4x4 for 32x32 input.
        # To be safe and general, we can use AdaptiveAvgPool2d(1) before the linear layer.
        # This handles any input size gracefully.
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'patch_size'}