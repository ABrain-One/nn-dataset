import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        
        # Store transform to satisfy requirement that it is read from prm and used (stored)
        self.transform = prm.get('transform', 'norm_32_flip')
        
        # Reduced channel widths to fit under 5M parameters
        # Stem: 3 -> 32
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet Blocks
        # Stage 1: 32 channels, 32x32, 2 blocks
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        # Stage 2: 64 channels, 16x16, 2 blocks
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        # Stage 3: 128 channels, 8x8, 2 blocks
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        # Stage 4: 256 channels, 4x4, 2 blocks
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        
        # Global Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(256, self.num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_planes, planes, blocks, stride):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1, None))
        return nn.Sequential(*layers)

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

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        lr = prm['lr']
        momentum = prm['momentum']
        batch_size = prm['batch']
        transform = prm['transform']
        
        # Ensure transform is explicitly used/stored
        self.transform = transform
        
        # Use SGD with momentum
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=5e-4
        )
        
        self.batch_size = batch_size
        self.base_lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, train_data):
        self.train()
        
        # Calculate total steps for scheduler
        total_steps = len(train_data)
        if total_steps == 0:
            return
            
        # Cosine annealing schedule
        for i, (inputs, labels) in enumerate(train_data):
            # Cosine schedule
            t = i / max(1, total_steps - 1)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * t))
            
            # Update optimizer LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()