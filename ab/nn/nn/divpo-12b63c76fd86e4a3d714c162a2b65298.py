import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        self.skip_add = stride == 1 and in_channels == out_channels
        self.out_channels = out_channels

    def forward(self, x):
        if not self.training and x.requires_grad:
            x = x.detach()
        if self.skip_add:
            x_skip = x
        else:
            x_skip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.skip_add:
            x = x + x_skip
        x = self.avg_pool(x)
        x = self.conv4(x)
        return x

class Block(nn.Module):
    def __init__(self, in_shape, out_channels, stride=1, expand_ratio=1):
        super().__init__()
        in_channels = in_shape[1]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            SimpleBlock(out_channels, out_channels, stride, expand_ratio),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.device = device
        self.blocks = Block(in_shape, 32, expand_ratio=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def forward(self, x):
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()