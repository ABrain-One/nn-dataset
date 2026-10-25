import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Derive input channels from in_shape for CIFAR-10
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            Block(in_channels, 32, stride=1),
            Block(32, 64, stride=2),
            Block(64, 128, stride=2)
        )
        self.layer1 = Block(128, 128, stride=1)
        self.layer2 = nn.Sequential(
            Block(128, 256, stride=2),
            Block(256, 256, stride=1)
        )
        self.layer3 = nn.Sequential(
            Block(256, 512, stride=2),
            Block(512, 512, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = None

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
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

    def __repr__(self):
        return "SimpleCNN"

def train_setup(self, prm):
    self.model.train_setup(prm)

def learn(self, train_data):
    self.model.learn(train_data)

def __call__(self, in_shape, out_shape, prm):
    self.device = torch.device('cpu')
    self.model = Net(in_shape, out_shape, prm, self.device)
    self._handler = Net(in_shape, out_shape, prm, self.device)
    return self.model

def __repr__(self):
    return "SimpleCNNTrainer()"

class SimpleCNNTrainer():
    def __init__(self):
        self._handler = None

    def __call__(self, in_shape, out_shape, prm):
        self.device = torch.device('cpu')
        self.model = Net(in_shape, out_shape, prm, self.device)
        self._handler = Net(in_shape, out_shape, prm, self.device)
        return self.model

    def as_trainer(self):
        return self

    def __repr__(self):
        return "SimpleCNNTrainer()"

model = SimpleCNNTrainer()