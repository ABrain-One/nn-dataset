from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torchvision.models._utils import _make_divisible


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
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
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

        # ResNet-9 configuration for CIFAR-10
        # Standard ResNet-9: 3 blocks of 128 channels
        # Structure: Conv(32) -> [Block(128), Block(128), Block(128)] -> Pool -> FC
        # Downsampling happens in the 2nd and 3rd blocks (strides 2, 2)
        # Input 32x32 -> Conv(16x16) -> Block1(16x16) -> Block2(8x8) -> Block3(4x4) -> Pool(1x1)

        self.in_planes = 64

        # Initial convolution
        # Standard ResNet starts with 64 channels
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet-9 has 9 layers (excluding conv1 and fc).
        # Usually implemented as 3 groups of 1 block each? Or 1 group of 3?
        # "ResNet-9" typically refers to a depth of 9.
        # Common CIFAR ResNet depths: 20, 56, 110.
        # ResNet-9 is a very shallow variant.
        # Let's look at the specific repo reference or standard interpretation.
        # A common "ResNet-9" for CIFAR is:
        # Conv -> BN -> ReLU
        # Block(64, 64, 1)
        # Block(64, 128, 2)
        # Block(128, 256, 2)
        # This is 3 blocks. Total conv layers: 1 (init) + 2*3 (blocks) = 7? No, 1+6=7.
        # Wait, "ResNet-9" usually implies 9 layers in the residual stack or total?
        # If it's 9 layers total:
        # Conv1 (1)
        # Block 1 (2)
        # Block 2 (2)
        # Block 3 (2)
        # Block 4 (2)
        # Block 5 (2)
        # Total 1 + 10 = 11?
        
        # Let's check the specific source "randyxia07/cifar10-resnet9".
        # Based on common shallow ResNets for CIFAR:
        # ResNet-9 often refers to:
        # 1 Conv layer
        # 4 Residual Blocks?
        # If we have 4 blocks, that's 8 layers in the stack.
        # Let's assume a standard configuration that fits "ResNet-9".
        # Often, ResNet-9 is defined as having 9 layers of convolution.
        # 1 (init) + 2 (block 1) + 2 (block 2) + 2 (block 3) + 2 (block 4) = 9?
        # If so, we have 4 blocks.
        
        # Let's try to match the "0.75 scale" mention if possible, but standard width is usually 64/128/256.
        # The prompt says "configurable width (e.g., 0.75 scale)".
        # However, without a specific hyperparameter for width, I should use the standard base width.
        # Standard ResNet base width is 64.
        
        # Let's implement a ResNet with 4 residual blocks to get close to 9 layers?
        # Or is it 3 blocks?
        # ResNet-20 is 3 blocks of 2 (1 + 2*3*2 = 13 layers? No, 1 + 6*2 = 13).
        # ResNet-9 is likely:
        # Conv1
        # Block 1 (64->64, stride 1)
        # Block 2 (64->128, stride 2)
        # Block 3 (128->256, stride 2)
        # This is 3 blocks. Total conv layers: 1 + 2 + 2 + 2 = 7.
        # Maybe the count includes BN or something else?
        
        # Let's look for "ResNet-9" specifically.
        # Some sources define ResNet-9 as:
        # 1x1 conv? No, CIFAR uses 3x3.
        
        # Let's assume the standard shallow ResNet configuration often cited for CIFAR which is compact.
        # If the source says "ResNet-9", and it's from randyxia07, let's look for clues.
        # A common shallow ResNet is ResNet-14 or ResNet-20.
        # ResNet-9 might be:
        # 1 Conv
        # 1 Block (64)
        # 1 Block (128)
        # 1 Block (256)
        # This is 3 blocks.
        
        # Let's try a configuration with 4 blocks to ensure depth?
        # If I use 4 blocks:
        # 1 (init) + 2 + 2 + 2 + 2 = 9 layers. This matches the name "ResNet-9" perfectly if we count conv layers.
        # So:
        # Conv1: 3->64
        # Block 1: 64->64, stride 1
        # Block 2: 64->128, stride 2
        # Block 3: 128->256, stride 2
        # Block 4: 256->512, stride 2? 
        # If stride 2 again, spatial size 32->32->16->8->4.
        # This seems reasonable.
        
        # However, many "ResNet-9" implementations for CIFAR use:
        # 1 Conv
        # 3 Blocks
        # And call it ResNet-9 because of some other counting method?
        
        # Let's stick to the most robust interpretation:
        # If the name is ResNet-9, and standard ResNet-N counts layers,
        # ResNet-20 has 1 + 3*2*2 = 13? No.
        # ResNet-20: 3 blocks, 2 layers each. 1 + 2*3*2 = 13 layers?
        # Actually, ResNet-20 is often called so because it has 20 layers in the original ImageNet paper context? No, CIFAR ResNets are named by total layers.
        # ResNet-20: 1 conv + 3 groups of 2 blocks = 1 + 6*2 = 13 conv layers?
        # Wait, the naming convention for CIFAR ResNets (He et al.) is:
        # ResNet-20: 3 blocks of 2.
        # ResNet-56: 9 blocks of 2.
        # ResNet-110: 18 blocks of 2.
        
        # So ResNet-9 would be?
        # If it follows the pattern, it's not a standard He et al. model.
        # It's likely a custom model.
        # Given "configurable width" and "compact", and the specific repo link, I will implement a ResNet with 4 residual blocks (total 9 conv layers: 1 init + 8 in blocks) to match the name numerically.
        
        # Configuration:
        # Block 1: 64 -> 64, stride 1
        # Block 2: 64 -> 128, stride 2
        # Block 3: 128 -> 256, stride 2
        # Block 4: 256 -> 512, stride 2
        
        # This results in 4x4 spatial size after the last block.
        # Adaptive pooling will handle it.

        self.layer1 = self._make_layer(64, 1)
        self.layer2 = self._make_layer(128, 1, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)
        self.layer4 = self._make_layer(512, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * BasicBlock.expansion, num_classes)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride))
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out