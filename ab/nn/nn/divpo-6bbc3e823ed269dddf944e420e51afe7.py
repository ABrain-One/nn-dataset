import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm
        self.channels = in_shape[1]
        self.stem = nn.Conv2d(self.channels, 32, 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        self.layers = nn.Sequential()
        self.init_layers(32, 32, 3, 1, 1, 'relu')
        self.init_layers(32, 64, 3, 1, 1, 'relu')
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Linear(64 * 6 * 6, 10)

    def init_layers(self, in_channels, out_channels, kernel, stride, padding, activation):
        block = nn.Sequential()
        block.add_module('conv2d', nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False))
        block.add_module('batchnorm', nn.BatchNorm2d(out_channels))
        if activation == 'relu':
            block.add_module('relu', nn.ReLU(inplace=True))
        elif activation == 'tanh':
            block.add_module('tanh', nn.Tanh())
        self.layers.append(block)
        return self.layers

    def forward(self, x):
        x = self.stem(x)
        x = self.stem_bn(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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