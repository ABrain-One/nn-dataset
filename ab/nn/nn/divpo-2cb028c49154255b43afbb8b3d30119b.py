import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return nn.ReLU(inplace=True)(out)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = out_shape[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block1 = Block(64, 64, stride=1)
        self.block2 = Block(64, 128, stride=2)
        self.block3 = Block(128, 128, stride=1)
        self.block4 = Block(128, 256, stride=2)
        self.block5 = Block(256, 256, stride=1)
        self.block6 = Block(256, 512, stride=2)
        self.block7 = Block(512, 512, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}
    def supported_hyperparameters(self):
        return {'lr', 'momentum'}
    def __repr__(self):
        return f"Net(in_channels={self.in_channels}, out_channels={self.num_classes}, device={self.device})"
    def supported_hyperparameters(self):
        return {'lr', 'momentum'}
    def __repr__(self):
        return f"Net(in_channels={self.in_channels}, out_channels={self.num_classes}, device={self.device})"
    def supported_hyperparameters(self):
        return {'lr', 'momentum'}
    def __repr__(self):
        return f"Net(in_channels={self.in_channels}, out_channels={self.num_classes}, device={self.device})"

def supported_hyperparameters():
    return {'lr', 'momentum'}

def train_setup(model, prm, device):
    model.to(device)
    model.criteria = (nn.CrossEntropyLoss().to(device),)
    model.optimizer = torch.optim.SGD(
        model.parameters(),
        lr=prm.get('lr', 0.01),
        momentum=prm.get('momentum', 0.9)
    )

def learn(model, train_data, device):
    model.train()
    for inputs, labels in train_data:
        inputs, labels = inputs.to(device), labels.to(device)
        model.optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.criteria[0](outputs, labels)
        loss.backward()
        model.optimizer.step()

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = out_shape[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block1 = Block(64, 64, stride=1)
        self.block2 = Block(64, 128, stride=2)
        self.block3 = Block(128, 128, stride=1)
        self.block4 = Block(128, 256, stride=2)
        self.block5 = Block(256, 256, stride=1)
        self.block6 = Block(256, 512, stride=2)
        self.block7 = Block(512, 512, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        train_setup(self, prm, self.device)

    def learn(self, train_data):
        learn(self, train_data, self.device)

    def __repr__(self):
        return f"Net(in_channels={self.in_channels}, out_channels={self.num_classes}, device={self.device})"

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}