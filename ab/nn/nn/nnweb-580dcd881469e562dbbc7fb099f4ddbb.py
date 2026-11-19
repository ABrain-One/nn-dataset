import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'dropout_aux'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.num_classes = out_shape[0]
        self.in_channels = in_shape[1] if len(in_shape) > 1 else 3
        
        self.dropout = prm['dropout']
        self.dropout_aux = prm['dropout_aux']
        
        # CIFAR-10 is 32x32. We use a Wide-ResNet style architecture.
        # Standard WRN-28-10 has ~36M params, which is too heavy for fast 1-epoch training and prone to overfitting on small data.
        # We use a lighter WRN-16-2 (depth 16, width 2) or similar.
        # Depth 16 means (16-2)/6 = 2 groups of blocks.
        # Width 2 means base width 16*2 = 32.
        
        depth = 16
        widen_factor = 2
        
        # Base channels
        base_width = 16 * widen_factor
        
        # Stem: 3 -> 16
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.blocks = nn.ModuleList()
        
        # Helper to create a block
        def make_block(in_planes, out_planes, stride):
            return BasicBlock(in_planes, out_planes, stride)
        
        # Group 1: stride 1
        self.blocks.append(make_block(16, base_width, 1))
        self.blocks.append(make_block(base_width, base_width, 1))
        
        # Group 2: stride 2
        self.blocks.append(make_block(base_width, base_width * 2, 2))
        self.blocks.append(make_block(base_width * 2, base_width * 2, 1))
        
        # Group 3: stride 2
        self.blocks.append(make_block(base_width * 2, base_width * 4, 2))
        self.blocks.append(make_block(base_width * 4, base_width * 4, 1))
        
        self.bn = nn.BatchNorm2d(base_width * 4)
        
        # Head
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(base_width * 4, self.num_classes)
        
        # Aux heads to use dropout_aux
        # We attach aux head to the output of the second group (base_width * 2)
        self.aux_head1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=self.dropout_aux),
            nn.Linear(base_width * 2, self.num_classes)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
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
            self.optimizer.step()
    
    def forward(self, x):
        x = self.stem(x)
        
        # Group 1
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        
        # Group 2
        x = self.blocks[2](x)
        x = self.blocks[3](x)
        
        # Aux 1 (Attached after Group 2, so input channels match base_width * 2)
        aux1 = self.aux_head1(x)
        
        # Group 3
        x = self.blocks[4](x)
        x = self.blocks[5](x)
        
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        x = self.dropout_layer(x)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out