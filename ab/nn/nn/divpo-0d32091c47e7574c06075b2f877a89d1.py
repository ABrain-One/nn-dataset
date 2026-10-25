import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Follow the overridden guidance for input channel extraction
        num_classes = int(out_shape[0])
        self.stem = Block(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = Block(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer2 = Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer3 = Block(128, 256, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer3(x)
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

    def __repr__(self):
        return f"Net(in_channels={self.stem.conv.in_channels}, hidden_size=256, num_classes={self.classifier.out_features})"

def train_setup(self, prm):
    self.model.to(self.device)
    self.criterion = nn.CrossEntropyLoss().to(self.device)
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=prm.get('lr', 0.01),
        momentum=prm.get('momentum', 0.9),
    )

def learn(self, train_data):
    self.model.train()
    for inputs, labels in train_data:
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

def __repr__(self):
    return f"TrainWrapper(model={self.model.__class__.__name__}, device={self.device})"

def supported_hyperparameters():
    return {'lr', 'momentum'}