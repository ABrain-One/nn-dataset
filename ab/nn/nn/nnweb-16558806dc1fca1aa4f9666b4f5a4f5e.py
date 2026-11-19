import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class ResidualBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions, batch norm, and ReLU."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
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
        dropout: float = prm['dropout']
        
        # Input shape: (C, H, W)
        # CIFAR-10 is 32x32. 
        # Stage 1: 3 -> 64 channels, stride 1 (no downsampling yet, or stride 2? 
        # "4 convolution stages reaching up to 256 channels"
        # Standard ResNet for CIFAR: 
        # Stem: Conv 3->64, BN, ReLU, Pool? 
        # Usually CIFAR ResNets don't use MaxPool after stem, they use stride 2 in first block.
        # Let's follow a standard CIFAR-10 ResNet layout adapted to "4 stages up to 256".
        
        # Stage 1: 64 channels
        # Stage 2: 128 channels
        # Stage 3: 256 channels
        # Stage 4: 256 channels (or 512? "up to 256" implies max is 256)
        
        # Let's assume:
        # Stem: Conv2d(3, 64, 3, padding=1), BN, ReLU
        # Stage 1: ResBlock(64, 64)
        # Stage 2: ResBlock(64, 128, stride=2)
        # Stage 3: ResBlock(128, 256, stride=2)
        # Stage 4: ResBlock(256, 256, stride=1) -- or maybe another 256 block?
        # "4 convolution stages"
        
        # If we start with 64, then 64, 128, 256, 256. That's 4 stages of blocks.
        # Or Stem + 3 stages? 
        # Let's do:
        # Stem: 3->64
        # Stage 1: 64->64
        # Stage 2: 64->128 (stride 2)
        # Stage 3: 128->256 (stride 2)
        # Stage 4: 256->256 (stride 1)
        
        # This results in spatial size:
        # Input: 32x32
        # Stem: 32x32
        # Stage 1: 32x32
        # Stage 2: 16x16
        # Stage 3: 8x8
        # Stage 4: 8x8
        
        # This is a reasonable interpretation.
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = ResidualBlock(64, 64, stride=1)
        self.stage2 = ResidualBlock(64, 128, stride=2)
        self.stage3 = ResidualBlock(128, 256, stride=2)
        self.stage4 = ResidualBlock(256, 256, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, out_shape[0])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x