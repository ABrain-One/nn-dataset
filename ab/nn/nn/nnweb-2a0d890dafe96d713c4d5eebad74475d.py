import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # ResNet-9 typically uses a width multiplier. 
        # "Configurable width (e.g., 0.75 scale)" suggests we might scale channels.
        # However, standard ResNet-9 on CIFAR often uses base width 64 or similar.
        # Let's stick to the standard "ResNet-9" definition from the referenced style (likely small depth).
        # A common ResNet-9 for CIFAR has 4 basic blocks (2 stages of 2 blocks each? Or 1 stage of 4? Or 2 stages of 2?).
        # "ResNet-9" usually implies depth 9.
        # Conv (1) + 4 Basic Blocks (4*2=8) + FC (1) = 10 layers? 
        # Or Conv (1) + 2 Basic Blocks (2*2=4) + ...?
        # Let's look at the specific repo "randyxia07/cifar10-resnet9".
        # In that repo, ResNet-9 typically consists of:
        # 1 Conv layer (3->64)
        # 4 Basic Blocks (64->64)
        # 1 FC layer
        # Total "layers" counted as convs+fc is 1+4+1 = 6? No, usually depth counts residual blocks.
        # If it's "ResNet-9", it likely has 9 layers total.
        # Conv (1) + 4 Basic Blocks (8 layers) + FC (1) = 10?
        # Maybe it's 1 Conv + 4 Blocks + FC = 6 modules, but depth 9?
        # Let's assume the standard small ResNet structure:
        # Stem: Conv 3->64, 3x3, pad 1
        # Stage 1: 4 Basic Blocks (64->64)
        # Head: Global Avg Pool -> FC
        
        # Width: The prompt mentions "configurable width (e.g., 0.75 scale)".
        # Since we don't have a 'width' hyperparameter in supported_hyperparameters, we must use a standard width.
        # Standard CIFAR ResNets often use 64 as the base width.
        
        base_width = 64
        
        # Stem
        self.conv1 = nn.Conv2d(in_shape[1], base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        
        # Blocks
        # ResNet-9: 4 basic blocks
        self.layer1 = self._make_layer(base_width, base_width, 4)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        dropout: float = prm['dropout']
        self.dropout = nn.Dropout(p=dropout)
        
        self.fc = nn.Linear(base_width, out_shape[0])
        
        self._init_weights()

    def _make_layer(self, in_planes, planes, num_blocks, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(in_planes, planes, stride if i == 0 else 1))
            in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out