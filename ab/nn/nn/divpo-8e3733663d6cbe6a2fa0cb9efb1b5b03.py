import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv1 = ConvBNRelu(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)
        self.conv2 = ConvBNRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups)
        self.downsample = ConvBNRelu(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups) if stride > 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.downsample(residual)
        return x + residual

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.make_layer(Block, 64, 64, stride=1, groups=1)
        self.layer2 = self.make_layer(Block, 64, 128, stride=2, groups=1)
        self.layer3 = self.make_layer(Block, 128, 256, stride=2, groups=1)
        self.layer4 = self.make_layer(Block, 256, 512, stride=2, groups=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def make_layer(self, block, in_channels, out_channels, stride, groups):
        layers = []
        layers.append(block(in_channels, out_channels, stride, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

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