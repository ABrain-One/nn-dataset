import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class ChannelOptimizedBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ChannelOptimizedHead(nn.Module):

    def __init__(self, in_channels, num_classes, prm):
        super().__init__()
        p = float(prm['dropout'])
        self.projector = nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dw_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256, bias=False)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(256, 32, kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(32, 256, kernel_size=1, bias=False), nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p)
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.projector(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dw_conv(x)
        x = x * self.se(x)
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
        self.layer1 = ChannelOptimizedBlock(3, 32, stride=1)
        self.layer2 = nn.Sequential(ChannelOptimizedBlock(32, 64, stride=2), ChannelOptimizedBlock(64, 64, stride=1), ChannelOptimizedBlock(64, 64, stride=1), ChannelOptimizedBlock(64, 64, stride=1))
        self.layer3 = nn.Sequential(ChannelOptimizedBlock(64, 128, stride=2), ChannelOptimizedBlock(128, 128, stride=1), ChannelOptimizedBlock(128, 128, stride=1), ChannelOptimizedBlock(128, 128, stride=1))
        self.layer4 = nn.Sequential(ChannelOptimizedBlock(128, 256, stride=2), ChannelOptimizedBlock(256, 256, stride=1), ChannelOptimizedBlock(256, 256, stride=1), ChannelOptimizedBlock(256, 256, stride=1))
        self.head = ChannelOptimizedHead(256, int(out_shape[0]), prm)
        self.to(device)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=float(prm['lr']), weight_decay=0.0001)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            loss = self.criteria(self(inputs), labels)
            loss.backward()
            self.optimizer.step()