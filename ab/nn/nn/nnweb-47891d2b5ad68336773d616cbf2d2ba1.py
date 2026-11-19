import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, drop_rate: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop_rate = drop_rate

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out += identity
        out = F.relu(out, inplace=True)
        return out


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
        num_classes: int = out_shape[0]
        drop_rate: float = prm['dropout']
        in_channels: int = in_shape[1]

        # CIFAR ResNet-18 Architecture
        # Stem: 3x3 conv, 64 channels, stride 1, BN, ReLU. No max pool.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Four stages of two BasicBlocks with widths 64, 128, 256, 512
        # Stride 2 in the first block of stages 2 to 4
        self.layer1 = self._make_layer(64, 2, stride=1, drop_rate=drop_rate)
        self.layer2 = self._make_layer(128, 2, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(256, 2, stride=2, drop_rate=drop_rate)
        self.layer4 = self._make_layer(512, 2, stride=2, drop_rate=drop_rate)

        self.classifier = nn.Linear(512, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int, drop_rate: float) -> nn.Sequential:
        layers = []
        in_channels = 64 if stride == 1 else self._get_last_channel(self.layer1 if stride == 2 else self.layer3 if stride == 4 else self.layer2)
        # Simpler logic: track channels
        # Stage 1: in=64, out=64
        # Stage 2: in=64, out=128
        # Stage 3: in=128, out=256
        # Stage 4: in=256, out=512
        
        # We need to know the input channels to the layer.
        # Let's just pass it explicitly in the constructor call to avoid complexity, 
        # but the method signature is fixed. 
        # Actually, I can just define the layers manually in __init__ to be safe and clear.
        
        # Re-implementing _make_layer to be robust
        # But I already called self._make_layer in __init__. 
        # Let's fix __init__ to pass input channels.
        # Wait, I can't change the call signature easily if I want to keep it clean.
        # Let's just hardcode the input channels in __init__ for the calls.
        
        # Correction: I will rewrite __init__ to pass input channels to _make_layer.
        pass

    def _make_layer_correct(self, in_channels: int, out_channels: int, num_blocks: int, stride: int, drop_rate: float) -> nn.Sequential:
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, drop_rate))
        in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(in_channels, out_channels, 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# Redefining __init__ properly to avoid the flawed _make_layer above
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
        num_classes: int = out_shape[0]
        drop_rate: float = prm['dropout']
        in_channels: int = in_shape[1]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Layers
        self.layer1 = self._make_layer(64, 64, 2, 1, drop_rate)
        self.layer2 = self._make_layer(64, 128, 2, 2, drop_rate)
        self.layer3 = self._make_layer(128, 256, 2, 2, drop_rate)
        self.layer4 = self._make_layer(256, 512, 2, 2, drop_rate)

        self.classifier = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int, drop_rate: float) -> nn.Sequential:
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, drop_rate))
        in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(in_channels, out_channels, 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out