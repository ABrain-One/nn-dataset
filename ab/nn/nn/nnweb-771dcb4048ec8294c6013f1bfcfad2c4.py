import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class CIFARResidualBlock(nn.Module):
    """
    Standard CIFAR Residual Block (Basic Block).
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+ shortcut) -> ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(CIFARResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class CIFARResidualNet(nn.Module):
    def __init__(self, in_channels, num_classes, depth=14):
        super(CIFARResidualNet, self).__init__()
        
        # Determine number of blocks per stage based on depth
        # For CIFAR-10, standard ResNet depths are 14, 26, 44, etc.
        # Depth = 2 * num_blocks + 2 (stem + final linear)
        # So num_blocks = (depth - 2) / 2
        if depth % 2 != 0:
            raise ValueError("Depth must be even for this implementation (e.g., 14, 26, 44)")
        num_blocks = (depth - 2) // 2
        
        # Standard CIFAR widths: 32, 64, 128
        widths = [32, 64, 128]

        # Stem: 3x3 Conv -> BN -> ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 32x32 resolution
        self.stage1 = self._make_layer(widths[0], widths[0], num_blocks, stride=1)
        
        # Stage 2: 16x16 resolution
        self.stage2 = self._make_layer(widths[0], widths[1], num_blocks, stride=2)
        
        # Stage 3: 8x8 resolution
        self.stage3 = self._make_layer(widths[1], widths[2], num_blocks, stride=2)

        # Head: Global Average Pooling and Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(widths[2], num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block may need downsampling
        layers.append(CIFARResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(CIFARResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]

        # Standard choice for "small residual network" on CIFAR-10 is ResNet-14
        # This corresponds to depth=14 (2 blocks per stage)
        self.model = CIFARResidualNet(self.channel_number, self.class_number, depth=14)
        
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.float()
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()