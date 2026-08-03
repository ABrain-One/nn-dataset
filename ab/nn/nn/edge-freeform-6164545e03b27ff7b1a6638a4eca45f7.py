import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

def supported_hyperparameters():
    return {'dropout', 'lr', 'momentum'}

class SEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EfficientMBConv(nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super(EfficientMBConv, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1, bias=False), nn.BatchNorm2d(in_channels * expand_ratio), nn.ReLU6(inplace=True), nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3, stride=1, padding=1, groups=in_channels * expand_ratio, bias=False), nn.BatchNorm2d(in_channels * expand_ratio), nn.ReLU6(inplace=True), SEBlock(in_channels * expand_ratio), nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Sequential() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.to(self.device)
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.dropout_rate = prm['dropout']
        self.features = nn.Sequential(nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU6(inplace=True), EfficientMBConv(32, 32), nn.MaxPool2d(kernel_size=2, stride=2), EfficientMBConv(32, 64), EfficientMBConv(64, 64), nn.MaxPool2d(kernel_size=2, stride=2), EfficientMBConv(64, 128), EfficientMBConv(128, 128), nn.MaxPool2d(kernel_size=2, stride=2), EfficientMBConv(128, 256), EfficientMBConv(256, 256), nn.MaxPool2d(kernel_size=2, stride=2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.classifier = nn.Linear(256, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def infer_dimensions_dynamically(self):
        dummy = torch.zeros(1, self.in_channels, 256, 256).to(self.device)
        dummy = self.features(dummy)
        dummy = self.adaptive_pool(dummy)
        dummy = torch.flatten(dummy, 1)
        return dummy.shape[1]

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()