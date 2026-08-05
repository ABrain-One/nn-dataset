import torch
import torch.nn as nn
from torch import Tensor

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SEBlock(nn.Module):

    def __init__(self, in_channels: int, reduction: int=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class InvertedResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int=1, se_reduction: int=16):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, in_planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes * self.expansion)
        self.conv2 = nn.Conv2d(in_planes * self.expansion, out_planes, kernel_size=3, stride=self.stride, padding=1, groups=in_planes * self.expansion, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.se = SEBlock(out_planes, se_reduction) if se_reduction else None
        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=self.stride, padding=0, bias=False), nn.BatchNorm2d(out_planes))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU6()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.se:
            out = self.se(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = nn.ReLU6()(out)
        return out

class CustomEfficientNet(nn.Module):

    def __init__(self, num_classes: int=10):
        super(CustomEfficientNet, self).__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU6())
        self.layer1 = self._make_layer(32, 32, 1, 1)
        self.layer2 = self._make_layer(32, 64, 2, 2)
        self.layer3 = self._make_layer(64, 128, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(6)
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, in_planes: int, out_planes: int, blocks: int, stride: int, se_reduction: int=16) -> nn.Sequential:
        layers = []
        layers.append(InvertedResidualBlock(in_planes, out_planes, stride, se_reduction))
        for _ in range(1, blocks):
            layers.append(InvertedResidualBlock(out_planes, out_planes, 1, se_reduction))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.backbone = CustomEfficientNet(num_classes=out_shape[0])
        self.backbone.to(device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.backbone.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def train_setup(self, prm):
        self.optimizer = torch.optim.SGD(self.backbone.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.backbone.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self.backbone(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 3.0)
            self.optimizer.step()