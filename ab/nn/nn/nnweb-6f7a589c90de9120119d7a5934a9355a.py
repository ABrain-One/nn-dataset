import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

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
            

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.num_classes = out_shape[0]
        self.in_channels = in_shape[1]
        
        # Hyperparameters
        # Using 'lr' and 'momentum' as required.
        # We assume standard values if not provided, but the constructor signature is fixed.
        # The 'prm' dict is passed to train_setup, but here we just use it to satisfy the "use" rule if needed, 
        # though typically hyperparameters are read in train_setup. 
        # However, the prompt says "each of those names ... must still be read from prm and actually used in the new model".
        # So I will read them here and potentially use them for initialization or architecture scaling if appropriate, 
        # or just store them. Let's store them to be safe, though they are primarily for the optimizer.
        self.lr = prm.get('lr', 0.1)
        self.momentum = prm.get('momentum', 0.9)

        # Architecture: Wide-ResNet style / ResNet-18 optimized for CIFAR-10
        # CIFAR-10 is 32x32. Standard ResNet uses 7x7 stride 2, which is too aggressive.
        # We will use a 3x3 stride 1 conv first.
        
        # Stem
        # Input: 3x32x32
        # Conv 3x3, stride 1, pad 1 -> 32x32
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # No MaxPool for CIFAR. We use stride 2 in the first block instead.

        # Blocks
        # Block 1: 64 -> 64 (stride 2)
        self.layer1 = self._make_layer(64, 64, 2, stride=2)
        # Block 2: 64 -> 128 (stride 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        # Block 3: 128 -> 256 (stride 2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        # Block 4: 256 -> 512 (stride 2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(self._basic_block(in_planes, planes, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(self._basic_block(planes, planes))
        return nn.Sequential(*layers)

    def _basic_block(self, in_planes, planes, stride=1, downsample=None):
        # Standard ResNet Basic Block
        conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(planes)
        conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(planes)
        
        block = nn.Sequential(
            conv1, bn1, nn.ReLU(inplace=True),
            conv2, bn2
        )
        
        # We need a custom forward to add the residual connection properly if downsample is used
        # Since we are returning a Sequential, we can't easily add the residual in __init__ without a custom class.
        # Let's define a proper BasicBlock class to be clean.

        # Actually, let's just define the class below and return an instance.
        # But _make_layer expects a module.
        # I will refactor to use a proper class.

        # Re-doing _make_layer to use a class.
        # See class definition below.
        pass

    # Let's replace the helper methods with a proper class definition to ensure correctness.

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Net(nn.Module):

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
            

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.num_classes = out_shape[0]
        self.in_channels = in_shape[1]
        
        # Read hyperparameters to satisfy constraints
        self.lr = prm.get('lr', 0.1)
        self.momentum = prm.get('momentum', 0.9)

        # CIFAR-10 specific ResNet-18
        # Stem: 3x3 conv, no maxpool
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Layers
        self.layer1 = self._make_layer(64, 64, 2, stride=2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(in_planes, planes, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes))
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