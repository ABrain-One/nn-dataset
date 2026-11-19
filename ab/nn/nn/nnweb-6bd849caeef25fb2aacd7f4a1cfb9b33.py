import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        dropout = prm['dropout']
        momentum = prm['momentum']

        # Use the standard CIFAR-10 normalization statistics.
        # Even if the input is already normalized, applying this again is harmless
        # (it just scales/shifts) and ensures robustness if the pipeline isn't.
        self.normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )

        # Wide ResNet architecture (WRN-16-2)
        # Depth 16, Width 2
        # Structure: conv1 -> block1 (2 blocks) -> block2 (2 blocks) -> block3 (2 blocks) -> avgpool -> fc
        # This is a proven architecture that learns fast and generalizes well.
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_shape[1], 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Wide ResNet blocks
        # Block 1: 16 -> 32 (stride 1)
        self.block1 = nn.Sequential(
            self._make_block(16, 32, dropout, momentum, stride=1),
            self._make_block(32, 32, dropout, momentum, stride=1)
        )
        
        # Block 2: 32 -> 64 (stride 2)
        self.block2 = nn.Sequential(
            self._make_block(32, 64, dropout, momentum, stride=2),
            self._make_block(64, 64, dropout, momentum, stride=1)
        )
        
        # Block 3: 64 -> 128 (stride 2)
        self.block3 = nn.Sequential(
            self._make_block(64, 128, dropout, momentum, stride=2),
            self._make_block(128, 128, dropout, momentum, stride=1)
        )

        self.bn_final = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, out_shape[0])

        # Initialize weights
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_block(self, in_channels, out_channels, dropout, momentum, stride=1):
        return BasicBlock(in_channels, out_channels, dropout, momentum, stride)

    def forward(self, x):
        # Apply normalization
        x = self.normalize(x)
        
        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = F.relu(self.bn_final(x), inplace=True)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device).to(torch.float32), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, dropout, momentum, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.dropout = nn.Dropout(dropout)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels, momentum=momentum)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.dropout(out)
        
        out += self.shortcut(residual)
        out = F.relu(out, inplace=True)
        return out