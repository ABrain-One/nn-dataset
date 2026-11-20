import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.num_classes = out_shape[0]
        self.in_channels = in_shape[1]

        # Use hyperparameters
        self.lr = prm['lr']
        self.momentum = prm['momentum']

        # Architecture: Wide ResNet-28-10 (WRN-28-10)
        # This is a standard high-performance architecture for CIFAR-10.
        # It learns fast and scales well.
        
        depth = 28
        widen_factor = 10
        
        # Calculate number of blocks
        assert (depth - 4) % 6 == 0, "Depth must be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        base_width = 64
        
        # Input stem
        self.conv1 = nn.Conv2d(self.in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        
        # Feature extraction layers
        # Each layer has 'n' blocks.
        # Layer 1: base_width
        # Layer 2: base_width * 2
        # Layer 3: base_width * 4
        
        self.layer1 = self._make_layer(base_width, n)
        self.layer2 = self._make_layer(base_width * 2, n, stride=2)
        self.layer3 = self._make_layer(base_width * 4, n, stride=2)
        
        self.bn2 = nn.BatchNorm2d(base_width * 4)
        
        # Classifier
        self.fc = nn.Linear(base_width * 4, self.num_classes)
        
        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(self._basic_block(self._get_current_width(), planes, stride))
            self._set_current_width(planes)
        return nn.Sequential(*layers)
    
    def _get_current_width(self):
        return getattr(self, '_current_width', 64)
        
    def _set_current_width(self, width):
        self._current_width = width
        
    def _basic_block(self, in_planes, out_planes, stride=1):
        return BasicBlock(in_planes, out_planes, stride)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

        self.batch_transform_fn = None  
        self.augment_configs = None

        if prm.get('augment'):          
            try:
                from ab.nn.transform.base.BatchTransform import batch_transform as bt
                self.batch_transform_fn = bt
                self.augment_configs = prm.get('augment')
                print(f"Augmentation loaded: {self.augment_configs}")  
            except ImportError as e:
                print(f"Failed to load batch_transform: {e}")

    def learn(self, train_data):
        total_batches = len(train_data)

        for batch_idx, (inputs, labels) in enumerate(train_data):  
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if self.batch_transform_fn: 
                # Pass batch index and total batches to find the fixed order

                inputs, target_a, target_b, lam = self.batch_transform_fn(
                    inputs, labels, batch_idx, total_batches, self.augment_configs  # fix: all 5 args
                )
                outputs = self(inputs)
                # Weighted loss calculation
                loss = lam * self.criteria[0](outputs, target_a) + (1. - lam) * self.criteria[0](outputs, target_b)
            else:
                outputs = self(inputs)
                loss = self.criteria[0](outputs, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
            
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