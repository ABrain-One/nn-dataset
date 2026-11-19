import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- MANDATORY FOR EVAL ENGINE ---
def supported_hyperparameters():
    return {'lr', 'momentum'}

# --- Basic ResNet Block ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, momentum=0.9):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, momentum=momentum)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

# --- Main Network ---
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device

        # in_shape is typically (C, H, W) or (B, C, H, W)
        # We need to extract C correctly.
        if len(in_shape) == 3:
            c_in = in_shape[0]
        elif len(in_shape) == 4:
            c_in = in_shape[1]
        else:
            c_in = 3 # Default to 3 (RGB)

        n_classes = out_shape[0] if out_shape else 10
        
        # Extract hyperparameters
        lr = prm['lr']
        momentum = prm['momentum']
        
        # Architecture: Lightweight ResNet (ResNet-18 structure with reduced channels)
        # Channels: 32, 64, 128, 256
        # This keeps param count well under 5M.
        
        self.in_planes = 32
        
        # Use the correctly extracted c_in
        self.conv1 = nn.Conv2d(c_in, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, momentum=momentum)
        
        self.layer1 = self._make_layer(BasicBlock, 32, 2, stride=1, momentum=momentum)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2, momentum=momentum)
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2, momentum=momentum)
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2, momentum=momentum)
        
        self.linear = nn.Linear(256 * BasicBlock.expansion, n_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride, momentum):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, momentum))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
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