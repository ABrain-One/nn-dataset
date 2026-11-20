import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_classes, in_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Stem: Conv 3x3, BN, ReLU, MaxPool
        # Using stride 1 in conv to keep spatial dimensions manageable for CIFAR (32x32)
        # Standard CIFAR ResNet uses 3x3 conv with stride 1 instead of 7x7 stride 2 + maxpool
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # CIFAR ResNet typically doesn't use maxpool in the stem to preserve resolution
        # But we can include a maxpool if we want, or just rely on strides in layers.
        # Let's stick to the standard CIFAR ResNet architecture (no maxpool in stem, stride 1 conv).
        
        # Layers:
        # Layer 1: 3 blocks, 64 channels, stride 1
        # Layer 2: 3 blocks, 128 channels, stride 2
        # Layer 3: 3 blocks, 256 channels, stride 2
        # Layer 4: 3 blocks, 512 channels, stride 2
        # This results in 32 -> 32 -> 16 -> 8 -> 4 spatial dimensions.
        
        self.layer1 = self._make_layer(block, 64, 3, stride=1)
        self.layer2 = self._make_layer(block, 128, 3, stride=2)
        self.layer3 = self._make_layer(block, 256, 3, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        # Read hyperparameters to ensure they are used
        self.lr = prm['lr']
        self.momentum = prm['momentum']
        
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]

        # ResNet-34 (or 18? ResNet-18 is 2 blocks per layer, ResNet-34 is 3 blocks per layer)
        # For CIFAR-10, ResNet-32 or ResNet-56 are common.
        # Let's use ResNet-34 architecture (3 blocks per layer) which is a good balance.
        # Actually, standard CIFAR ResNet papers use 20, 32, 44, 56, 110, 1202.
        # ResNet-32 is 3 blocks per layer (1+3+3+3+1 layers? No, 1 stem + 3 layers of 3 blocks + 1 fc = 32 conv layers).
        # My _make_layer creates 3 blocks. 4 layers * 3 blocks = 12 blocks + 1 stem = 13 conv layers?
        # Wait, standard ResNet-34 has 34 layers.
        # Let's stick to the implementation above which creates 3 blocks per layer.
        # Total conv layers: 1 (stem) + 3 + 3 + 3 + 3 = 13 conv layers?
        # Each BasicBlock has 2 convs.
        # Stem: 1 conv.
        # Layer 1: 3 blocks * 2 = 6 convs.
        # Layer 2: 3 blocks * 2 = 6 convs.
        # Layer 3: 3 blocks * 2 = 6 convs.
        # Layer 4: 3 blocks * 2 = 6 convs.
        # Total: 1 + 6 + 6 + 6 + 6 = 25 conv layers. This is roughly ResNet-26 or so.
        # It's a solid, fast-learning architecture.
        
        self.model = ResNet(BasicBlock, self.class_number, in_channels=self.channel_number)

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm['momentum'], 0.999), weight_decay=1e-4)
        self.max_batches = prm.get('max_batches', None)
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            if self.max_batches is not None and i >= self.max_batches: break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
