import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard ResNet Basic Block for CIFAR (3x3 convs, no downsampling by default)."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop_rate = drop_rate

        # Shortcut projection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out += self.shortcut(residual)
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
        drop_rate = prm['dropout']

        # ResNet-9 Configuration for CIFAR-10 (32x32)
        # Structure: Stem + 3 groups of 1 block each = 3 residual blocks total.
        # Standard CIFAR ResNet-9 uses [1, 1, 1] blocks with widths [64, 128, 256].
        # Downsampling happens between groups 1->2 and 2->3 to reduce spatial size from 32 to 4.
        
        # Widths
        width_1 = 64
        width_2 = 128
        width_3 = 256

        # Stem
        # CIFAR ResNets typically use a single 3x3 conv as stem, no max pool.
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], width_1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width_1),
            nn.ReLU(inplace=True)
        )

        # Group 1: 1 block, stride 1 (no downsampling)
        self.layer1 = self._make_layer(width_1, width_1, num_blocks=1, stride=1, drop_rate=drop_rate)
        
        # Group 2: 1 block, stride 2 (downsample 32->16)
        self.layer2 = self._make_layer(width_1, width_2, num_blocks=1, stride=2, drop_rate=drop_rate)
        
        # Group 3: 1 block, stride 2 (downsample 16->8)
        # Note: Some implementations go to 4x4, but 8x8 is common for ResNet-9/11 variants 
        # or the final pool handles it. Given "compact" and "avoid aggressive downsampling",
        # we stick to standard CIFAR depth. 
        # Wait, ResNet-9 is 9 layers. Stem(1) + 3x(2) = 9.
        # If we downsample twice, we end up at 8x8. 
        # Let's verify standard ResNet-9 CIFAR structure.
        # Usually: Conv(32->32), Block(32->32), Block(32->64, stride 2), Block(64->128, stride 2).
        # This results in 8x8 features.
        self.layer3 = self._make_layer(width_2, width_3, num_blocks=1, stride=2, drop_rate=drop_rate)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(width_3, num_classes)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, drop_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_channels, out_channels, stride, drop_rate))
            in_channels = out_channels * BasicBlock.expansion
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}