import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Head(nn.Module):

    def __init__(self, in_channels, num_classes, prm):
        super().__init__()
        p = float(prm['dropout'])
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(512, 512 // 16, kernel_size=1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(512 // 16, 512, kernel_size=1, bias=True), nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p)
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        se_weights = self.se(x)
        x = x * se_weights
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.dropout = float(prm['dropout'])
        self.batch = int(prm['batch'])
        self.learning_rate = float(prm['lr'])
        self.momentum_value = float(prm['momentum'])
        self.num_epochs = int(prm['epoch'])
        self._transform_name = prm['transform']
        self.stem = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer3 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.layer4 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.head = Head(512, int(out_shape[0]), prm)
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        lr = float(prm['lr'])
        momentum = float(prm['momentum'])
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def learn(self, train_data):
        self.train()
        batch_hint = self.batch
        for _epoch in range(max(1, self.num_epochs)):
            for inputs, labels in train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criteria(outputs, labels)
                loss.backward()
                self.optimizer.step()