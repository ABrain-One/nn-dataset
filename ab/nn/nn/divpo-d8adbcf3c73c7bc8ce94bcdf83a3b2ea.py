import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = Block(in_shape[1], 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = Block(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer2 = Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer3 = Block(128, 256, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])
        self.to(device)

        self.criteria = (nn.CrossEntropyLoss().to(device),)
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# The provided in_shape is (batch, channels, height, width) for CIFAR-10.
def get_in_channels(in_shape):
    return in_shape[1]

def supported_hyperparameters():
    return {'lr', 'momentum'}

def train_setup(self, prm):
    self.net.train_setup(prm)

def learn(self, train_data):
    self.net.learn(train_data)

def __next__(self):
    raise StopIteration

class DataLoaderWrapper:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.data_loader)
        return data[0].to(self.device), data[1]

class CIFAR10DataModule:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                    transform=torchvision.transforms.ToTensor())
        self.data_loader = DataLoaderWrapper(torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True))

    def train_dataloader(self):
        return self.data_loader

class Learner:
    def __init__(self, net, prm, device):
        self.net = net
        self.device = device
        self.prm = prm

    def fit(self, epochs, dataloader):
        self.net.train()
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.net.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.net.criteria[0](outputs, labels)
                loss.backward()
                self.net.optimizer.step()

class TrainModule(nn.Module):
    def __init__(self, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm
        self.net = Net(in_shape=(3, 32, 32), out_shape=(10,), prm=prm, device=device)
        self.learner = Learner(self.net, prm, device)

    def forward(self, x):
        return self.net(x)

    def train_setup(self, prm):
        self.net.train_setup(prm)

    def learn(self, train_data):
        self.learner.fit(epochs=1, dataloader=train_data)

    def learn_epochs(self, epochs, train_data):
        self.learner.fit(epochs, train_data)

def supported_hyperparameters():
    return {'lr', 'momentum'}

def train_setup(self, prm):
    self.net.train_setup(prm)

def learn(self, train_data):
    self.learner.fit(epochs=1, dataloader=train_data)

def learn_epochs(self, epochs, train_data):
    self.learner.fit(epochs, train_data)

def supported_hyperparameters():
    return {'lr', 'momentum'}

def train_setup(self, prm):
    self.net.train_setup(prm)

def learn(self, train_data):
    self.learner.fit(epochs=1, dataloader=train_data)

def learn_epochs(self, epochs, train_data):
    self.learner.fit(epochs, train_data)

def train_setup(self, prm):
    self.net.train_setup(prm)

def learn(self, train_data):
    self.learner.fit(epochs=1, dataloader=train_data)

def learn_epochs(self, epochs, train_data):
    self.learner.fit(epochs, train_data)

def train_setup(self, prm):
    self.net.train_setup(prm)

def learn(self, train_data):
    self.learner.fit(epochs=1, dataloader=train_data)

def learn_epochs(self, epochs, train_data):
    self.learner.fit(epochs, train_data)

def train_setup(self, prm):
    self.net.train_setup(prm)

def learn(self, train_data):
    self.learner.fit(epochs=1, dataloader=train_data)

def learn_epochs(self, epochs, train_data):
    self.learner.fit(epochs, train_data)

def train_setup(self, prm):
    self.net.train_setup(prm)

def learn(self, train_data):
    self.learner.fit(epochs=1, dataloader=train_data)

def learn_epochs(self, epochs, train_data):
    self.learner.fit(epochs, train_data)

def train_setup(self, prm):
    self.net.train_setup(prm)