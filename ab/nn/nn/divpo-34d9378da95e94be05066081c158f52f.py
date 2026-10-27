import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.stem = nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 64, kernel_size=3, stride=1, padding=1)
        self.block2 = Block(64, 128, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = None

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

params = 500000

# Auto-generated edge-friendly shape (channels-first C x H x W)
in_shape = [3, 32, 32]
out_shape = [10]
device = torch.device('cpu')
prm = {'lr': 0.01, 'momentum': 0.9}

# Standard import to avoid circular imports
if 'Block' not in globals():
    from torch import nn

# Target max param count: 500000
# Keep product of channels (C) * H * W <= 450000
# Use standard layers only, favor reducing H or W over C
# Derive conv1_channels and conv2_channels from in_channels <= 32
# Resulting shape should be as close as possible to (64, 32)
# Target edge-friendly latency after shape computation

# Auto-generated block for edge-friendly shape
class AutoGenBlock(nn.Module):
    def __init__(self, in_channels):
        super(AutoGenBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))

    def get_out_shape(self):
        batch, channels, height, width = in_shape
        x = torch.zeros(batch, channels, height, width, device=device)
        x = self(x)
        return x.shape[1], x.shape[2], x.shape[3]

# Target shape close to (64, 32)
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1 = AutoGenBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, out_shape[0])

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

def supported_hyperparameters():
    return {'lr', 'momentum'}