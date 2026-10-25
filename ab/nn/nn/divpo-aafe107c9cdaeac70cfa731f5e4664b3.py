import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.in_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=0.0001)

    def build_features(self):
        layers = []
        layers += [
            Block(self.in_channels, 32, kernel_size=3, stride=2, dilation=1),
            Block(32, 32, kernel_size=3, stride=1, dilation=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            Block(32, 32, kernel_size=3, stride=1, dilation=4),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            Block(32, 32, kernel_size=3, stride=1, dilation=8),
            Block(32, 32, kernel_size=3, stride=1, dilation=16),
            Block(32, 32, kernel_size=3, stride=1, dilation=32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 32, kernel_size=3, stride=1, dilation=32),
            Block(32, 32, kernel_size=3, stride=1, dilation=16),
            Block(32, 32, kernel_size=3, stride=1, dilation=8),
            Block(32, 32, kernel_size=3, stride=1, dilation=4),
            Block(32, 32, kernel_size=3, stride=1, dilation=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 32, kernel_size=3, stride=1, dilation=2),
            Block(32, 32, kernel_size=3, stride=1, dilation=4),
            Block(32, 32, kernel_size=3, stride=1, dilation=8),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'], weight_decay=0.0001)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()