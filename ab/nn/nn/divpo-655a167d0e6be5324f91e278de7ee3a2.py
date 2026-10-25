import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(in_channels, out_channels, kernel_size, stride, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, padding=1):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            conv_bn(in_channels, in_channels * expand_ratio, kernel_size, stride, padding),
            conv_bn(in_channels * expand_ratio, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.conv1 = conv_bn(in_shape[1], 32, 3, 2, 1)
        self.b1 = Block(32, 32, 3, 1, 1)
        self.b2 = Block(32, 32, 3, 1, 6)
        self.b3 = Block(32, 32, 3, 1, 6)
        self.b4 = Block(32, 64, 3, 2, 6)
        self.b5 = Block(64, 64, 3, 1, 6)
        self.b6 = Block(64, 64, 3, 1, 6)
        self.b7 = Block(64, 64, 3, 1, 6)
        self.b8 = Block(64, 96, 3, 2, 6)
        self.b9 = Block(96, 96, 3, 1, 6)
        self.b10 = Block(96, 96, 3, 1, 6)
        self.b11 = Block(96, 96, 3, 1, 6)
        self.b12 = Block(96, 160, 3, 2, 6)
        self.b13 = Block(160, 160, 3, 1, 6)
        self.b14 = Block(160, 160, 3, 1, 6)
        self.b15 = Block(160, 160, 3, 1, 6)
        self.b16 = Block(160, 320, 3, 2, 6)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(320, out_shape[0])
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        x = self.b7(x)
        x = self.b8(x)
        x = self.b9(x)
        x = self.b10(x)
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)
        x = self.b15(x)
        x = self.b16(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

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