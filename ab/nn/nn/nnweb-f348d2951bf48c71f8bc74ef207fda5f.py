import torch
import torch.nn as nn
from typing import cast, Dict, List, Union


def make_layers(in_channels, cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 256, "M", 512, 512, 512, 256, "M"],
}


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


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


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
        num_classes: int = out_shape[0]
        dropout: float = prm['dropout']
        
        # Input: 32x32x3 (CIFAR-10 standard, derived from in_shape)
        # Stage 1: 3 -> 64, stride 1, size 32x32
        # Stage 2: 64 -> 128, stride 2, size 16x16
        # Stage 3: 128 -> 256, stride 2, size 8x8
        # Stage 4: 256 -> 256, stride 1, size 8x8
        
        # The description says "4 convolution stages reaching up to 256 channels".
        # It also mentions "Residual-style skip connection in deeper feature stages".
        # A common interpretation for CIFAR-10 CNNs with this description (like the one from the source) is a VGG-style or ResNet-style block.
        # Given "up to 256", let's use widths: 64, 128, 256, 256.
        # Downsampling: usually happens between stages.
        # Let's construct:
        # Conv1: 3->64, BN, ReLU
        # Conv2: 64->64, BN, ReLU
        # Pool: MaxPool 2x2 -> 16x16
        # ResBlock1: 64->128 (stride 2) -> 8x8
        # ResBlock2: 128->256 (stride 1) -> 8x8
        # ResBlock3: 256->256 (stride 1) -> 8x8
        
        # Wait, the source "cifar10-from-scratch-cnn" by 11Cpss typically uses a specific architecture.
        # Looking at the quote: "4 convolution stages (up to 256 channels) - Batch normalization after convolution blocks - Residual-style skip connection in deeper feature stages"
        
        # Let's implement a ResNet-like structure adapted to CIFAR-10.
        # Standard ResNet for CIFAR uses stride 1 in first layer, then stride 2 in subsequent stages.
        # Stage 1: 3->64, 3x3, pad 1, stride 1. Output 32x32.
        # Stage 2: 64->128, 3x3, pad 1, stride 2. Output 16x16.
        # Stage 3: 128->256, 3x3, pad 1, stride 2. Output 8x8.
        # Stage 4: 256->256, 3x3, pad 1, stride 1. Output 8x8.
        
        # The prompt asks for "4 convolution stages".
        # I will implement a sequence of blocks.
        # To satisfy "Residual-style skip connection in deeper feature stages", I will use ResidualBlocks for the later stages.
        # A simple interpretation:
        # Layer 1: Conv 3->64, BN, ReLU
        # Layer 2: Conv 64->64, BN, ReLU
        # Pool: MaxPool 2x2
        # Layer 3: ResBlock 64->128 (stride 2)
        # Layer 4: ResBlock 128->256 (stride 2)
        # Layer 5: ResBlock 256->256 (stride 1)
        
        # This gives 4 "stages" if we group them? Or just 4 conv layers?
        # "4 convolution stages" usually implies 4 blocks of convolutions.
        # Let's stick to a clean ResNet-18 style but adapted to CIFAR dimensions and max 256 channels.
        
        # Block 1: 3->64 (stride 1)
        # Block 2: 64->128 (stride 2)
        # Block 3: 128->256 (stride 2)
        # Block 4: 256->256 (stride 1)
        
        # This matches "up to 256".
        
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 2: 64 -> 128
        self.layer2 = ResidualBlock(64, 128, stride=2)
        
        # Stage 3: 128 -> 256
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        # Stage 4: 256 -> 256
        self.layer4 = ResidualBlock(256, 256, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes)
        )
        
        # Dropout is required by hyperparameters. I will add it before the classifier.
        self.dropout = nn.Dropout(p=dropout)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x