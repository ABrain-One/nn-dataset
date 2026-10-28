import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential()
        self.features.add_module('conv1', nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('bn1', nn.BatchNorm2d(64))
        self.features.add_module('relu1', nn.ReLU(inplace=True))
        self.features.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module('layer1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('bn2', nn.BatchNorm2d(128))
        self.features.add_module('relu2', nn.ReLU(inplace=True))
        self.features.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module('layer2', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('bn3', nn.BatchNorm2d(256))
        self.features.add_module('relu3', nn.ReLU(inplace=True))
        self.features.add_module('pool3', nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module('layer3', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('bn4', nn.BatchNorm2d(512))
        self.features.add_module('relu4', nn.ReLU(inplace=True))
        self.features.add_module('pool4', nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module('layer4', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('bn5', nn.BatchNorm2d(512))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        self.features.add_module('flatten', nn.Flatten())
        self.classifier = nn.Linear(512, out_shape[0])
        self.to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()