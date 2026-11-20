import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


# Basic Residual Block for CIFAR (no downsampling)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# ResNet-9 Architecture
# Structure:
# Stem: 3x3 Conv -> BN -> ReLU
# Stage 1: 3 ResBlocks (stride 1)
# Stage 2: 3 ResBlocks (stride 2)
# Stage 3: 3 ResBlocks (stride 2)
# Total blocks: 9
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, width_scale=1.0):
        super(ResNet9, self).__init__()
        
        # Base width for ResNet-18 is 64. 
        # ResNet-9 typically uses a smaller base width for efficiency on CIFAR.
        # A common base width for small ResNets on CIFAR is 32 or 64.
        # Let's use base_width = 32 * width_scale. 
        # If width_scale=0.75, base_width = 24.
        # However, to keep it simple and standard, let's define the widths explicitly 
        # based on a base of 32 (common for CIFAR ResNets) scaled by width_scale.
        
        base_width = int(32 * width_scale)
        
        # Ensure minimum width
        if base_width < 16:
            base_width = 16
            
        w1 = base_width
        w2 = base_width * 2
        w3 = base_width * 4
        
        # Stem
        self.conv1 = nn.Conv2d(in_channels, w1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w1)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1: 3 blocks, stride 1
        self.layer1 = self._make_layer(ResidualBlock, w1, w1, num_blocks=3, stride=1)
        
        # Stage 2: 3 blocks, stride 2
        self.layer2 = self._make_layer(ResidualBlock, w1, w2, num_blocks=3, stride=2)
        
        # Stage 3: 3 blocks, stride 2
        self.layer3 = self._make_layer(ResidualBlock, w2, w3, num_blocks=3, stride=2)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w3, num_classes)
        
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers.append(block(in_channels, out_channels, strides[0]))
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, strides[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.channel_number = in_shape[1]
        self.class_number = out_shape[0]
        
        # The description mentions a configurable width (e.g., 0.75 scale).
        # Since 'width' is not in supported_hyperparameters, we use a standard default.
        # For CIFAR, a width scale of 0.75 with base 32 is reasonable, or simply base 32.
        # Let's use width_scale=1.0 (base 32) as a standard compact choice, or 0.75 as hinted.
        # The prompt says "choose the standard choice". ResNet-9 often implies a specific compact config.
        # Let's stick to base_width=32 (scale 1.0) for simplicity and robustness, or 24 (0.75).
        # Given "configurable width (e.g., 0.75 scale)", I will implement it with a default of 0.75 
        # to match the "compact" description, but since I can't read 'width' from prm, I hardcode the scale.
        # Let's use 0.75 as it's explicitly cited in the description as an example.
        
        self.model = ResNet9(self.channel_number, self.class_number, width_scale=0.75)
        
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