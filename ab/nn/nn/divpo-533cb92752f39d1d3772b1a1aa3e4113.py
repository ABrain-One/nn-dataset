import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1)
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build_layer(128, 128, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.criteria = nn.CrossEntropyLoss()
        for p in self.parameters():
            nn.init.normal_(p, mean=0, std=0.1)

    def build_layer(self, in_channels, out_channels, stride, groups):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
        )

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

def check_shape(net, input_shape):
    x = torch.zeros(input_shape)
    x = net(x)
    assert isinstance(x, torch.Tensor)
    assert x.ndim == 1, 'output should be a 1D tensor'

def supported_hyperparameters():
    return {'lr', 'momentum'}

def train_setup(net, prm):
    net.train_setup(prm)

def learn(net, train_data):
    net.learn(train_data)

def net_creator(in_shape, out_shape, prm, device):
    net = Net(in_shape, out_shape, prm, device)
    net_size = 0
    for v in net.parameters():
        net_size += v.element_size() * v.nelement()
    assert net_size <= 500000, 'Total parameter count exceeds the limit.'
    input_shape = (in_shape[1], 32, 32) if in_shape[0] == 3 else (32, 32)
    check_shape(net, input_shape)
    return net