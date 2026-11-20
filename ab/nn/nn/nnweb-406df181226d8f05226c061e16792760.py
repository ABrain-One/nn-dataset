import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- MANDATORY FOR EVAL ENGINE ---
def supported_hyperparameters():
    return {'lr', 'momentum'}

# --- Efficient Block (BasicBlock) ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, momentum=0.75, drop_path_prob=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, momentum=momentum)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# --- Main Network ---
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device

        # Determine input size from in_shape (assuming [C, H, W] or [B, C, H, W])
        # CIFAR-10 is 32x32. We derive H, W to be safe.
        if len(in_shape) == 3:
            _, H, W = in_shape
        else:
            _, _, H, W = in_shape
            
        self.in_channels = 3
        self.num_classes = out_shape[0] if out_shape else 10
        
        # Hyperparameters
        self.momentum = prm.get('momentum', 0.9)
        
        # Architecture: ResNet-18 style but with slightly wider channels to fit param budget and improve capacity
        # Standard ResNet-18 is ~11.2M. We need < 5M.
        # Let's use a ResNet-18 with base width 64 (standard) but maybe reduce depth or width?
        # Actually, ResNet-18 with base 64 is the target. Let's check params.
        # Stem: 3->64 (3*64*7*7 = 9408)
        # Layer 1: 2 blocks of 64. (64->64)
        # Layer 2: 2 blocks of 128. (64->128)
        # Layer 3: 2 blocks of 256. (128->256)
        # Layer 4: 2 blocks of 512. (256->512)
        # FC: 512->10
        # This is the standard 11.2M.
        
        # To get under 5M, we can use a "Small" ResNet.
        # Let's try Base Width 64, but fewer layers? Or smaller width?
        # If we use width 32:
        # Stem: 3->32
        # L1: 32->32
        # L2: 32->64
        # L3: 64->128
        # L4: 128->256
        # FC: 256->10
        # This should be well under 5M and very fast.
        
        base_w = 32
        num_blocks = [2, 2, 2, 2] # ResNet-18 depth structure

        self.in_planes = base_w
        
        # Stem: Standard ResNet uses 7x7 stride 2. For CIFAR (32x32), 7x7 stride 2 is often replaced by 3x3 stride 1 to preserve resolution longer, or just 3x3 stride 1.
        # Let's use 3x3 stride 1 for CIFAR to keep 32x32 -> 32x32.
        self.conv1 = nn.Conv2d(self.in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, momentum=self.momentum)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(base_w, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_w * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_w * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_w * 8, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_w * 8, self.num_classes)
        
        self.to(device)

    def _make_layer(self, plane, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, plane, stride, momentum=self.momentum))
            self.in_planes = plane * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def train_setup(self, prm):
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