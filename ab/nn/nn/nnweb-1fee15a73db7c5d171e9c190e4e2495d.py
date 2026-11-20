import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_se:
            out = self.se(out)

        out += identity
        out = self.relu(out)
        return out


class ResNetSE(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, depth=20, base_width=16):
        super(ResNetSE, self).__init__()
        
        # For CIFAR, we use a small stem to preserve spatial resolution
        self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        
        # Determine number of blocks per stage based on depth
        # Standard ResNet depth: (depth - 2) / 6
        # For depth 20: 2 blocks per stage
        # For depth 32: 3 blocks per stage
        # For depth 56: 5 blocks per stage
        n = (depth - 2) // 6
        
        self.layer1 = self._make_layer(base_width, n, stride=1, use_se=True)
        self.layer2 = self._make_layer(base_width * 2, n, stride=2, use_se=True)
        self.layer3 = self._make_layer(base_width * 4, n, stride=2, use_se=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(base_width * 4, num_classes)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride, use_se):
        layers = []
        layers.append(ResBlock(self.conv1.out_channels if not hasattr(self, 'layer1') else self._get_prev_channels(), out_channels, stride, use_se))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1, use_se))
        return nn.Sequential(*layers)

    def _get_prev_channels(self):
        # This helper is a bit hacky, let's just track channels manually in __init__
        # Actually, let's rewrite __init__ to be cleaner.
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Let's refine the ResNetSE to be cleaner and ensure channel tracking is correct.
class ResNetSE_Clean(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, depth=20, base_width=16):
        super(ResNetSE_Clean, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        
        n = (depth - 2) // 6
        
        self.layer1 = self._make_layer(base_width, base_width, n, stride=1, use_se=True)
        self.layer2 = self._make_layer(base_width, base_width * 2, n, stride=2, use_se=True)
        self.layer3 = self._make_layer(base_width * 2, base_width * 4, n, stride=2, use_se=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_width * 4, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride, use_se):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, use_se))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1, use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]
        
        # ResNet-20 with SE blocks, base width 16
        # This is a strong baseline for CIFAR-10
        self.model = ResNetSE_Clean(self.channel_number, self.class_number, depth=20, base_width=16)

        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()