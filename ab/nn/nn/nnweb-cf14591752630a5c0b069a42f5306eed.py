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

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.transform = prm.get('transform', 'norm_32_flip')

        num_classes = out_shape[0] if isinstance(out_shape, (tuple, list)) else out_shape

        # Architecture: ResNet-like with reduced width to stay under 5M parameters.
        # Channels: 32 -> 64 -> 128 -> 256
        # Blocks: 2 per stage
        
        # Stem: 3 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Layer 1: 32 -> 32 (2 blocks)
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        
        # Layer 2: 32 -> 64 (2 blocks)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        
        # Layer 3: 64 -> 128 (2 blocks)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        # Layer 4: 128 -> 256 (2 blocks)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

        self._init_weights()

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
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.device = next(self.parameters()).device
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with momentum. Momentum in prm is used directly.
        # Weight decay is set to 0 to avoid conflict with standard SGD momentum usage in this context.
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=0
        )
        
        # Store batch size for potential scheduler adjustments if needed.
        self.batch_size = prm.get('batch', 128)
        
        # Estimate total steps. CIFAR-10 has 50,000 training images.
        self.total_steps = None
        self.current_step = 0
        self.initial_lr = prm['lr']

    def learn(self, train_data):
        self.train()
        
        # Determine total steps for cosine schedule
        if hasattr(train_data, '__len__'):
            total_steps = len(train_data)
        else:
            total_steps = 50000 // self.batch_size
            
        self.total_steps = total_steps
        self.current_step = 0
        
        # Precompute cosine schedule multipliers
        # Cosine annealing: lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
        
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Update learning rate
            if self.total_steps > 0:
                t = self.current_step / self.total_steps
                t = min(1.0, max(0.0, t))
                multiplier = 0.5 * (1 + math.cos(math.pi * t))
                current_lr = self.initial_lr * multiplier
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            self.current_step += 1
            # Stop if we've done enough steps (in case dataloader is longer than expected)
            if self.current_step >= self.total_steps:
                break