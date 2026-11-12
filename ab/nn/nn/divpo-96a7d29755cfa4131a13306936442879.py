import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, expansion=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU6(inplace=True)
        self.skip = (in_channels == out_channels) and (stride == 1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.skip:
            x += residual
        x = self.act2(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1)
        self.b1 = Block(32, 32, 3, 1, 1)
        self.b2 = Block(32, 32, 3, 1, 1)
        self.b3 = Block(32, 64, 3, 1, 1, 1.5)
        self.b4 = Block(64, 64, 3, 1, 1)
        self.b5 = Block(64, 64, 3, 1, 1)
        self.b6 = Block(64, 64, 3, 1, 2, 1.5)
        self.b7 = Block(64, 64, 3, 1, 1)
        self.b8 = Block(64, 64, 3, 1, 1)
        self.b9 = Block(64, 96, 3, 1, 1, 1.5)
        self.b10 = Block(96, 96, 3, 1, 1)
        self.b11 = Block(96, 96, 3, 1, 1)
        self.b12 = Block(96, 96, 3, 1, 2, 1.5)
        self.b13 = Block(96, 96, 3, 1, 1)
        self.b14 = Block(96, 96, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(96 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.pool(x)
        x = self.b6(x)
        x = self.b7(x)
        x = self.b8(x)
        x = self.b9(x)
        x = self.b10(x)
        x = self.b11(x)
        x = self.pool(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)
        x = self.pool(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

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

def supported_hyperparameters():
    return {'lr', 'momentum'}