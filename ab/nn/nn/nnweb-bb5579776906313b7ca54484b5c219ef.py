import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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
        out += residual
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        
        # ResNet-9 configuration
        # Total depth 9: Stem (1) + 4 BasicBlocks (4*2=8 convs? No, ResNet-9 is usually defined by number of blocks)
        # Standard ResNet-9 for CIFAR:
        # Conv stem
        # Block 1 (stride 1)
        # Block 2 (stride 2)
        # Block 3 (stride 2)
        # Total layers: 1 (stem) + 2 (block1) + 2 (block2) + 2 (block3) = 7 convs? 
        # Wait, "ResNet-9" typically refers to a network with 9 layers of weight.
        # Common ResNet-9 definition:
        # 1 Conv (stem)
        # 1 BasicBlock (2 convs)
        # 1 BasicBlock (2 convs)
        # 1 BasicBlock (2 convs)
        # Total: 1 + 2 + 2 + 2 = 7? No.
        # Let's look at the source "cifar10-resnet9".
        # Often ResNet-9 is:
        # Conv
        # Block
        # Block
        # Block
        # Total layers: 1 + 2 + 2 + 2 = 7? 
        # Actually, many implementations of ResNet-9 for CIFAR use:
        # Stem: Conv
        # Layer 1: 1 Block
        # Layer 2: 1 Block
        # Layer 3: 1 Block
        # This is often called ResNet-9 because 1 (stem) + 2*3 (blocks) = 7? No.
        # Let's check the depth calculation.
        # ResNet-18 is 18 layers.
        # ResNet-9 is likely 1 stem + 4 blocks? Or 1 stem + 2 blocks?
        # If we follow the "ResNet-9" name from the specific repo linked (randyxia07/cifar10-resnet9):
        # The repo typically implements a small resnet.
        # Let's assume the standard small ResNet structure:
        # 1 Stem Conv
        # 3 Blocks of 2 convs each? That would be 7 layers.
        # Maybe it means 9 total layers including stem?
        # 1 (stem) + 2 (block1) + 2 (block2) + 2 (block3) + 2 (block4)? That's 9.
        # So: 1 Stem + 4 BasicBlocks.
        # Width: "configurable width (e.g., 0.75 scale)". Standard base width is 64.
        # 0.75 * 64 = 48.
        # Let's use width_mult = 0.75.
        # Base channels: 64.
        # Channels: 48, 96, 192?
        # Or maybe just fixed channels.
        # Let's stick to a standard interpretation:
        # Stem: 3 -> 64 (or scaled)
        # Block 1: 64 -> 64, stride 1
        # Block 2: 64 -> 128, stride 2
        # Block 3: 128 -> 256, stride 2
        # If we scale by 0.75:
        # Stem: 3 -> 48
        # Block 1: 48 -> 48, stride 1
        # Block 2: 48 -> 96, stride 2
        # Block 3: 96 -> 192, stride 2
        # Total convs: 1 (stem) + 2 + 2 + 2 = 7. This is often called ResNet-7 or similar.
        
        # Let's look for "ResNet-9" specifically.
        # Some sources define ResNet-9 as:
        # Conv
        # Block
        # Block
        # Block
        # Block
        # Total 1 + 2*4 = 9 layers.
        # Let's implement 4 blocks.
        
        width_mult = 0.75
        base_channels = 64
        channels = [int(base_channels * width_mult), int(base_channels * 2 * width_mult), int(base_channels * 4 * width_mult)]
        # channels = [48, 96, 192]
        
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = self._make_layer(channels[0], channels[0], stride=1, num_blocks=1)
        self.layer2 = self._make_layer(channels[0], channels[1], stride=2, num_blocks=1)
        self.layer3 = self._make_layer(channels[1], channels[2], stride=2, num_blocks=2)
        
        # Total blocks: 1 + 1 + 2 = 4 blocks.
        # Layers: 1 (stem) + 2 + 2 + 4 = 9 layers. This fits "ResNet-9".
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2], self.num_classes)

    def _make_layer(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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