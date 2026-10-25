import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1)
        self.layer1 = self.build_layer(32, 32, stride=1)
        self.layer2 = self.build_layer(32, 64, stride=2)
        self.layer3 = self.build_layer(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        num_classes = out_shape[0]
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.to(device)
        self.criteria = (nn.CrossEntropyLoss(),)
        self.optimizer = self.configure_optimizers(prm)

    def build_layer(self, in_channels, out_channels, stride):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def configure_optimizers(self, prm):
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        optimizer_type = prm.get('optimizer', 'SGD')
        if optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=momentum
            )
        elif optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
            )
        elif optimizer_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=lr,
                momentum=momentum
            )
        return optimizer

def check_shape(shape):
    assert len(shape) in {1, 2}, shape
    assert shape[0] in {1, 3}, shape[0]
    assert all(300 >= x >= 64 for x in shape[1:]), shape[1:]

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.conv1 = Block(in_shape[1], 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = Block(64, 192, 3, 1, 1)
        self.conv3 = Block(192, 192, 3, 1, 1)
        self.conv4 = Block(192, 192, 3, 1, 1)
        self.conv5 = Block(192, 320, 3, 2, 1)
        self.conv6 = Block(320, 320, 3, 1, 1)
        self.conv7 = Block(320, 320, 3, 1, 1)
        self.conv8 = Block(320, 320, 3, 1, 1)
        self.conv9 = Block(320, 320, 3, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(320 * 6 * 6, out_shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()