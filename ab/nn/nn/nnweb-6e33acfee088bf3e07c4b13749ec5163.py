import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_p=0.0):
        super().__init__()
        self.drop = nn.Dropout2d(drop_p)
        self.conv1 = ConvBNReLU(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.drop(x)
        out = self.conv1(out)
        out = self.drop(out)
        out = self.conv2(out)
        # Use out.add_(shortcut) instead of out += shortcut to avoid in-place modification
        # of the tensor that is an output of a previous op (ReluBackward0).
        # Actually, `out` is the result of self.conv2(out), which is a new tensor.
        # The error says "output 0 of ReluBackward0".
        # Let's trace:
        # out = self.drop(x)
        # out = self.conv1(out) -> ReLU(BN(Conv))
        # out = self.drop(out)
        # out = self.conv2(out) -> ReLU(BN(Conv))
        # The error mentions ReluBackward0.
        # If we do `out += shortcut`, we modify `out` in-place.
        # If `out` is required for gradient computation of a previous op, this fails.
        # Using `out = out + shortcut` creates a new tensor, which is safe.
        out = out + self.shortcut(identity)
        return F.relu(out)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        # Use dropout hyperparameter for regularization (0.0 if not present in prm)
        drop_p = prm.get('dropout', 0.0)

        # ResNet-style architecture
        # CIFAR-10 is 32x32. We want to downsample to 4x4 before global pooling.
        # 32 -> 16 -> 8 -> 4. 
        # Standard ResNet uses strides at the start of stages 2, 3, 4.
        # Stage 1: 32x32 (stride 1)
        # Stage 2: 16x16 (stride 2)
        # Stage 3: 8x8 (stride 2)
        # Stage 4: 4x4 (stride 2)
        
        # We will use a depth of 18 or 34 for better performance. Let's go with ResNet-18 equivalent.
        # Blocks: 2, 2, 2, 2
        # Channels: 64, 128, 256, 512
        
        # Stem
        # Standard ResNet uses 7x7 stride 2 for ImageNet. For CIFAR, we use 3x3 stride 1.
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # No maxpool for CIFAR to preserve spatial info, or use stride in first block?
        # Common CIFAR ResNet:
        # Layer 1: 64 channels, 32x32
        # Layer 2: 128 channels, 16x16
        # Layer 3: 256 channels, 8x8
        # Layer 4: 512 channels, 4x4
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1, drop_p=drop_p)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, drop_p=drop_p)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, drop_p=drop_p)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, drop_p=drop_p)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, self.num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, drop_p):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, drop_p))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, drop_p))
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
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

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