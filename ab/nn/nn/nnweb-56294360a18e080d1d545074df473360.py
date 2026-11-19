import torch
import torch.nn as nn
from torch import Tensor


def supported_hyperparameters():
    return {'lr', 'momentum'}


class CIFARResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(CIFARResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.act1 = activation(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.act2 = activation(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.act2(out)
        
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
        num_classes = out_shape[0]
        
        # Configuration
        # Standard CIFAR-10 ResNet-20 configuration: 3 stages, 2 blocks each, widths 64, 128, 256
        base_width = 64
        num_blocks = [2, 2, 2]
        strides = [1, 2, 2]
        
        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU
        
        # Hyperparameters
        lr = prm['lr']
        momentum = prm['momentum']
        
        # Stem: 3x3 conv, no stride (standard for CIFAR)
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], base_width, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(base_width),
            activation(inplace=True)
        )
        
        # Stages
        # Stage 1: 64 -> 64, stride 1
        self.stage1 = self._make_stage(base_width, base_width, num_blocks[0], strides[0], norm_layer, activation)
        # Stage 2: 64 -> 128, stride 2
        self.stage2 = self._make_stage(base_width, base_width * 2, num_blocks[1], strides[1], norm_layer, activation)
        # Stage 3: 128 -> 256, stride 2
        self.stage3 = self._make_stage(base_width * 2, base_width * 4, num_blocks[2], strides[2], norm_layer, activation)
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_width * 4, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, norm_layer, activation):
        layers = []
        # First block handles stride and channel change
        layers.append(CIFARResBlock(in_channels, out_channels, stride, norm_layer, activation))
        in_channels = out_channels
        # Remaining blocks have stride 1
        for _ in range(1, num_blocks):
            layers.append(CIFARResBlock(in_channels, out_channels, 1, norm_layer, activation))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x