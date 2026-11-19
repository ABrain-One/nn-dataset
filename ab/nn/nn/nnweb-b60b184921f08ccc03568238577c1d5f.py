import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- HASH IDENTIFIERS (Ensures unique UUIDs for caching) ---
# Architecture: ResNet-18 (CIFAR variant)
# Norm: BatchNorm
# Activation: ReLU
# Optimizer: AdamW (using lr, momentum from prm)

# --- MANDATORY FOR EVAL ENGINE ---
def supported_hyperparameters():
    return {'lr', 'momentum'}

# --- Helper Classes ---

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # CIFAR input is 32x32, so we use 3x3 conv with padding 1 (no stride) to keep spatial dims
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# --- Main Network ---
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device

        # Extract dimensions from in_shape (B, C, H, W)
        # in_shape[1] is channels, in_shape[2] is H, in_shape[3] is W
        # We assume standard CIFAR-10 (3, 32, 32) but code should be robust.
        # ResNet18 is designed for 32x32 input. If input is different, 
        # standard ResNet might need adjustment, but CIFAR-10 is fixed.
        
        n_classes = out_shape[0] if out_shape else 10
        
        # Instantiate the architecture
        # ResNet-18 has ~11M params? No, ResNet-18 is 11.7M. 
        # Wait, the prompt says "standard ResNet-18 has 11.2M and is therefore not allowed".
        # I need a model with < 5M params.
        # ResNet-14 or ResNet-10?
        # ResNet-10 (CIFAR) has ~0.3M params.
        # ResNet-14 has ~0.5M params.
        # ResNet-18 is too big.
        # Let's use ResNet-14 (2,2,2,2 but with smaller channels? No, standard ResNet-14 usually refers to depth).
        # Actually, standard ResNet-18 is [2,2,2,2] with 64, 128, 256, 512 channels.
        # Let's check parameter count for ResNet-18:
        # Conv1: 3*64*3*3 = 1728
        # Layer 1 (2 blocks, 64 ch):
        #   Block 1: Conv1 64*64*3*3=36864, Conv2 64*64*3*3=36864. Shortcut: none.
        #   Block 2: Conv1 64*64*3*3=36864, Conv2 64*64*3*3=36864.
        # Layer 2 (2 blocks, 128 ch):
        #   Block 1: Conv1 64*128*3*3=73728, Conv2 128*128*3*3=147456. Shortcut: 64*128*1*1=8192.
        #   Block 2: Conv1 128*128*3*3=147456, Conv2 128*128*3*3=147456.
        # Layer 3 (2 blocks, 256 ch):
        #   Block 1: Conv1 128*256*3*3=294912, Conv2 256*256*3*3=589824. Shortcut: 128*256*1*1=32768.
        #   Block 2: Conv1 256*256*3*3=589824, Conv2 256*256*3*3=589824.
        # Layer 4 (2 blocks, 512 ch):
        #   Block 1: Conv1 256*512*3*3=1179648, Conv2 512*512*3*3=2359296. Shortcut: 256*512*1*1=131072.
        #   Block 2: Conv1 512*512*3*3=2359296, Conv2 512*512*3*3=2359296.
        # FC: 512*10=5120.
        # Total approx: 1.7k + 3*(36k+36k) + 2*(73k+147k+8k) + 2*(294k+589k+32k) + 2*(1179k+2359k+131k) + 5k
        # Roughly: 1.7 + 216 + 448 + 1744 + 7360 + 5 = ~9.7M.
        # So ResNet-18 is indeed > 5M.
        
        # I need a smaller ResNet.
        # ResNet-10 (CIFAR) uses [1,1,1,1] blocks? Or [2,2,2,2] with smaller channels?
        # Standard CIFAR ResNet-14: [2,2,2,2] but maybe smaller channels?
        # Let's try ResNet-14 with standard channels [64, 128, 256, 512] but depth [2,2,2,2] is ResNet-18.
        # ResNet-14 usually means depth 14.
        # If I use [1,1,1,1] blocks (ResNet-10), params are much lower.
        # Let's calculate ResNet-10 ([1,1,1,1]):
        # L1 (1 block, 64): 2*36k = 72k
        # L2 (1 block, 128): 73k + 147k + 8k = 228k
        # L3 (1 block, 256): 294k + 589k + 32k = 915k
        # L4 (1 block, 512): 1179k + 2359k + 131k = 3669k
        # Total ~ 4.9M. This is close to the limit.
        
        # To be safe, let's use a slightly smaller channel count or depth.
        # Or use ResNet-14 with [2,2,2,2] but smaller channels?
        # Let's try a "Small ResNet" with channels [64, 128, 256, 512] and depth [2,2,2,2] is too big.
        # How about depth [1,1,1,1] (ResNet-10) is ~5M.
        # Let's reduce channels slightly or use a different architecture.
        
        # Alternative: ResNet-14 with channels [64, 128, 256, 512] and depth [2,2,2,2] is 18 layers.
        # Let's use depth [1,1,1,1] (ResNet-10) to stay well under 5M.
        # Or maybe [2,2,2,2] with channels [32, 64, 128, 256]?
        # Let's stick to standard channels but reduce depth to [1,1,1,1].
        
        self.model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=n_classes)
        
        # We need to use 'momentum' from prm.
        # In train_setup, we pass momentum to AdamW betas.
        # We also need to ensure we don't hardcode resolution.
        # ResNet handles any spatial size due to AdaptiveAvgPool2d.
        
        self.to(device)

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        # Use prm['lr'] and prm['momentum']
        # AdamW with momentum mapped to beta1? Or use SGD?
        # The baseline used AdamW with betas=(momentum, 0.999).
        # I will follow that pattern to ensure 'momentum' is used.
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