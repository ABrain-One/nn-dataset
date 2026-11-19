import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def supported_hyperparameters():
    return {'lr', 'momentum'}


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
        self.lr = prm['lr']
        self.momentum = prm['momentum']

        num_classes = out_shape[0]
        in_channels = in_shape[1]

        # ResNet-9 Configuration
        # Total blocks: 4 in first group (stride 1), 4 in second group (stride 2)
        # Standard ResNet-9 for CIFAR often uses [BasicBlock, BasicBlock, BasicBlock, BasicBlock]
        # with channels [64, 128, 256] or similar.
        # The description mentions "configurable width (e.g., 0.75 scale)".
        # Standard base widths for ResNet are often [64, 128, 256].
        # Let's use the standard small ResNet-9 structure:
        # Conv1 (7x7 or 3x3? For CIFAR 32x32, usually 3x3) -> BN -> ReLU
        # No MaxPool (to preserve spatial info as per description "avoids aggressive downsampling in early layers")
        # Layer1: 4 blocks, 64 channels, stride 1
        # Layer2: 4 blocks, 128 channels, stride 2
        # Total blocks = 8. Plus initial conv?
        # ResNet-18 is 18 layers (2+2+2+2 blocks * 2 convs = 16 + 1 conv + 1 fc).
        # ResNet-9 implies 9 layers? Or 9 blocks?
        # The source "cifar10-resnet9" by randyxia07 typically refers to a ResNet with 9 layers of convolutional operations or a specific depth.
        # A common "ResNet-9" for CIFAR is actually a ResNet with 9 convolutional layers (excluding FC).
        # Structure:
        # Conv1 (1)
        # Block 1 (2 convs)
        # Block 2 (2 convs)
        # Block 3 (2 convs)
        # Block 4 (2 convs)
        # Total 9 conv layers.
        # This matches the "ResNet-9" naming convention where depth = number of conv layers.
        # So we need 4 residual blocks.
        # To avoid aggressive downsampling, we can keep stride 1 for the first few blocks.
        # Standard CIFAR ResNet downsamples at layer 2 and 3.
        # If we have 4 blocks, we can do:
        # Block 1: stride 1, 64 ch
        # Block 2: stride 1, 64 ch (or 128?)
        # Block 3: stride 2, 128 ch
        # Block 4: stride 2, 256 ch?
        
        # Let's look at the specific repo structure if possible.
        # randyxia07/cifar10-resnet9:
        # Usually defines:
        # self.in_planes = 64
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.layer1 = self._make_layer(64, 2, stride=1)
        # self.layer2 = self._make_layer(128, 2, stride=2)
        # self.layer3 = self._make_layer(256, 2, stride=2)
        # That would be 1 + 2*2 + 2*2 + 2*2 = 1 + 12 = 13 conv layers. That's ResNet-18 equivalent.
        
        # Wait, "ResNet-9" might mean 9 layers total including FC? No.
        # Let's assume the standard small ResNet for CIFAR that is often called "ResNet-9" in some contexts is actually a ResNet with 9 *convolutional* layers.
        # Conv1 (1) + 4 Blocks (8 convs) = 9 conv layers.
        # This fits the description "compact".
        
        # Configuration:
        # Conv1: 3 -> 64, 3x3, stride 1
        # Block 1: 64 -> 64, 3x3, stride 1
        # Block 2: 64 -> 64, 3x3, stride 1
        # Block 3: 64 -> 128, 3x3, stride 2
        # Block 4: 128 -> 128, 3x3, stride 2
        # This gives 1 + 2 + 2 + 2 + 2 = 9 conv layers.
        
        # However, some "ResNet-9" implementations use width multiplier.
        # Let's stick to standard widths [64, 128, 256] but adjust block count to match "9".
        # If I use 4 blocks, I get 9 conv layers.
        
        # Let's implement this structure.
        
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        # 4 blocks total to reach 9 conv layers (1 init + 4*2 blocks)
        # To avoid aggressive downsampling, keep first blocks at stride 1.
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        # Wait, if I use 2 blocks in layer 1 and 2 blocks in layer 2, that's 4 blocks.
        # 1 (conv1) + 4 (layer1) + 4 (layer2) = 9 conv layers.
        # This seems correct for a "ResNet-9".
        
        self.linear = nn.Linear(self.in_planes, num_classes)

    def _make_layer(self, plane, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, plane, stride))
            self.in_planes = plane * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out