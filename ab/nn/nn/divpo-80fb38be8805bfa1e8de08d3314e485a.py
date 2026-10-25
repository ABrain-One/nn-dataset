import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.block1 = Block(64, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 128)
        self.block4 = Block(128, 256)
        self.block5 = Block(256, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def supported_hyperparameters():
    return {'lr', 'momentum'}


def prm_from_device(device):
    return {'lr': 0.01, 'momentum': 0.9}


def train_setup(model, prm):
    device = prm.get('device', torch.device('cpu'))
    model.to(device)
    model.criteria = (nn.CrossEntropyLoss().to(device),)
    model.optimizer = torch.optim.SGD(
        model.parameters(),
        lr=prm.get('lr', 0.01),
        momentum=prm.get('momentum', 0.9),
    )
    return model


def learn(model, train_data):
    model.train()
    for inputs, labels in train_data:
        inputs = inputs.to(model.device)
        labels = labels.to(model.device)
        model.optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.criteria[0](outputs, labels)
        loss.backward()
        model.optimizer.step()