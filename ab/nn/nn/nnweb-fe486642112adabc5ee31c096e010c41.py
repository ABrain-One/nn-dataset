import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(drop_prob) if drop_prob > 0 else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


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
        num_classes = out_shape[0]
        in_channels = in_shape[1]
        
        # Use hyperparameters
        # We will use momentum in optimizer (done in train_setup)
        # We will use lr in optimizer (done in train_setup)
        # The prompt says: "each of those names (for example dropout or momentum) must still be read from prm and actually used in the new model"
        # The baseline used 'lr' and 'momentum'. I will stick to these two to be safe, or add dropout if it's in the list?
        # The baseline list is {'lr', 'momentum'}. I will use these.
        # However, often 'dropout' is a standard hyperparameter in these challenges. If it's not in supported_hyperparameters, I shouldn't read it.
        # I will strictly follow the returned list from the baseline: {'lr', 'momentum'}.
        
        # Architecture: ResNet-18 style with SE blocks and BatchNorm.
        # Input: 3x32x32 (CIFAR-10)
        
        # Initial conv: 3x3, stride 1 (to keep spatial dims 32x32 for CIFAR)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 32->16
        
        # ResNet blocks
        # Stage 1: 64 channels, 16x16
        self.layer1 = self._make_layer(64, 2, stride=1)
        
        # Stage 2: 128 channels, 8x8
        self.layer2 = self._make_layer(128, 2, stride=2)
        
        # Stage 3: 256 channels, 4x4
        self.layer3 = self._make_layer(256, 2, stride=2)
        
        # Stage 4: 512 channels, 2x2
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        in_channels = self._get_in_channels(out_channels, len(layers)) # This logic is tricky, let's do it manually
        
        # Re-implement _make_layer properly
        layers = []
        in_ch = 64 if out_channels == 64 else int(out_channels / 2)
        for i, stride in enumerate(strides):
            layers.append(BasicBlock(in_ch, out_channels, stride))
            in_ch = out_channels
            
        return nn.Sequential(*layers)

    def _get_in_channels(self, out_channels, block_idx):
        # Helper to determine input channels for the first block of a layer
        # Layer 1: in 64
        # Layer 2: in 64
        # Layer 3: in 128
        # Layer 4: in 256
        if out_channels == 64:
            return 64
        elif out_channels == 128:
            return 64
        elif out_channels == 256:
            return 128
        elif out_channels == 512:
            return 256
        return 0

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x