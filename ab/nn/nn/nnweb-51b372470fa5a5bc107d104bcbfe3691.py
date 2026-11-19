import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-9.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + Shortcut -> ReLU
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.downsample = None
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet9(nn.Module):
    """
    ResNet-9 architecture for CIFAR-10.
    Layout:
    - Initial Conv 3x3 (stride 1)
    - 3 groups of 2 residual blocks each (total 6 blocks)
    - Widths: 64, 128, 256
    - Downsampling: Stride 1 in first group, Stride 2 in subsequent groups
    """
    def __init__(self, in_channels, num_classes, width_scale=1.0):
        super(ResNet9, self).__init__()
        
        # Calculate widths based on scale
        # Standard ResNet-18 base widths are 64, 128, 256, 512.
        # For ResNet-9 (3 groups), we use 64, 128, 256.
        # Applying width_scale.
        base_widths = [64, 128, 256]
        widths = [int(w * width_scale) for w in base_widths]
        
        # Ensure widths are multiples of 8 for efficient computation, though not strictly required
        # Standard choice: just use the calculated integers.
        
        self.in_planes = 64

        # Initial convolution
        # CIFAR-10 uses 3x3 stride 1 conv first, unlike ImageNet which uses 7x7 stride 2
        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Layers
        # Group 1: 2 blocks, no downsampling (stride 1)
        self.layer1 = self._make_layer(widths[0], 2, stride=1)
        
        # Group 2: 2 blocks, downsampling (stride 2)
        self.layer2 = self._make_layer(widths[1], 2, stride=2)
        
        # Group 3: 2 blocks, downsampling (stride 2)
        self.layer3 = self._make_layer(widths[2], 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[2] * BasicBlock.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, out_channels, stride))
            self.in_planes = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]
        
        # ResNet-9 typically uses a width scale. 
        # The prompt mentions "configurable width (e.g., 0.75 scale)".
        # We will use 1.0 as the standard base unless specified otherwise, 
        # but to be safe and align with "standard choice", we use 1.0 (64, 128, 256).
        # However, many "ResNet-9" implementations for CIFAR use smaller widths like 64, 128, 256 is standard ResNet-18/34 style.
        # Let's stick to the standard ResNet-18 width profile for the first 3 stages: 64, 128, 256.
        
        self.model = ResNet9(self.channel_number, self.class_number, width_scale=1.0)

    def forward(self, x):
        return self.model(x)

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