import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(32, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def _make_layer(self, inplanes, planes, blocks):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}..{self.num_classes} -> {self.count_parameters():,} parameters)"


def prm_create():
    return {'lr': 0.1, 'momentum': 0.9}


def supported_hyperparameters():
    return {'lr', 'momentum'}


def check_device(device):
    return device.type != 'cpu' and device.memory_allocated() < 8589934592


def train_setup(net, prm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.train()
    net.optimizer = torch.optim.SGD(
        net.parameters(),
        lr=prm.get('lr', 0.01),
        momentum=prm.get('momentum', 0.9),
    )
    return net, device


def learn(net, train_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    for inputs, labels in train_data:
        inputs, labels = inputs.to(device), labels.to(device)
        net.optimizer.zero_grad()
        outputs = net(inputs)
        loss = net.criteria(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 3)
        net.optimizer.step()
    return net


def net_create(in_shape, out_shape, prm):
    class Net(nn.Module):
        def __init__(self, in_shape, out_shape, prm, device):
            super().__init__()
            self.device = device
            self.in_channels = in_shape[1]
            self.num_classes = out_shape[0]
            self.learning_rate = prm['lr']
            self.momentum = prm['momentum']

            self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.layer1 = self._make_layer(32, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2)
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

            self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

        def _make_layer(self, inplanes, planes, blocks):
            layers = []
            layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

            return x

        def train_setup(self, prm):
            self.to(self.device)
            self.criteria = nn.CrossEntropyLoss().to(self.device)
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
                loss = self.criteria(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 3)
                self.optimizer.step()

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def __repr__(self):
            return f"{self.__class__.__name__}({self.in_channels}..{self.num_classes} -> {self.count_parameters():,} parameters)"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(in_shape, out_shape, prm, device)
    net.train_