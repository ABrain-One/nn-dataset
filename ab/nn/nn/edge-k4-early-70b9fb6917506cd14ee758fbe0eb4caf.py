import torch
import torch.nn as nn
import torchvision.transforms as transforms

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SqueezeExcite(nn.Module):

    def __init__(self, in_channels, reduction_ratio=4):
        super(SqueezeExcite, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Block(nn.Module):

    def __init__(self, inplanes, planes, stride=1, reduction=4):
        super(Block, self).__init__()
        self.expansion = 1
        self.conv1 = nn.Conv2d(inplanes, inplanes * self.expansion, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(inplanes * self.expansion, reduction)
        self.conv2 = nn.Conv2d(inplanes * self.expansion, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.se(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(32, 32, 3, 1)
        self.layer2 = self._make_layer(32, 64, 4, 2)
        self.layer3 = self._make_layer(64, 128, 6, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, out_shape[0])
        self.to(self.device)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, inplanes, planes, blocks, stride):
        layers = []
        layers.append(Block(inplanes, planes, stride))
        for _ in range(1, blocks):
            layers.append(Block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, data):
        self.train()
        for inputs, targets in data:
            inputs, targets = (inputs.to(self.device), targets.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()