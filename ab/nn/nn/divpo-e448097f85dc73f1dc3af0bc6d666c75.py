import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, expand_ratio=1, drop_rate=0.):
        super().__init__()
        self.drop_rate = drop_rate
        self.use_residual = in_channels == out_channels and stride == 1
        self._channels = in_channels

        _channels = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, _channels, kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(_channels),
            nn.ReLU6(inplace=True),
            nn.Dropout(drop_rate),
            nn.Conv2d(_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes

        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.block1 = Block(32, 32, expand_ratio=1, drop_rate=0.2)
        self.block2 = Block(32, 32, expand_ratio=2, drop_rate=0.3)
        self.block3 = Block(32, 64, stride=2, expand_ratio=4, drop_rate=0.4)
        self.block4 = Block(64, 64, expand_ratio=2, drop_rate=0.3)
        self.block5 = Block(64, 64, expand_ratio=4, drop_rate=0.4)
        self.block6 = Block(64, 96, stride=2, expand_ratio=4, drop_rate=0.4)
        self.block7 = Block(96, 96, expand_ratio=2, drop_rate=0.3)
        self.block8 = Block(96, 96, expand_ratio=4, drop_rate=0.4)
        self.block9 = Block(96, 96, expand_ratio=2, drop_rate=0.3)
        self.block10 = Block(96, 96, expand_ratio=4, drop_rate=0.4)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flat = nn.Flatten()
        self.classifier = nn.Linear(96 * 6 * 6, num_classes)

        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.pool(x)
        x = self.flat(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

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