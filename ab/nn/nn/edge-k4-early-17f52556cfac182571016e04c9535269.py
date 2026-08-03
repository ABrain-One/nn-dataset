import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SqueezeAndExcite(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(SqueezeAndExcite, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BlockWithSE(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BlockWithSE, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SqueezeAndExcite(planes * self.expansion, reduction=16)
        self.downsample = []
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * self.expansion))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if len(self.downsample) > 0:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.layer1 = self._make_layer(32, 32, 2, 1)
        self.layer2 = self._make_layer(32, 64, 2, 1)
        self.layer3 = self._make_layer(64, 128, 2, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, out_shape[0])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = []
        layers.append(BlockWithSE(in_planes, planes, stride))
        for _ in range(1, blocks):
            layers.append(BlockWithSE(planes, planes))
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
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()