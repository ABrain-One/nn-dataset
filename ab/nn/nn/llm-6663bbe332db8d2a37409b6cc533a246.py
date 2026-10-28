import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential()
        self.classifier = nn.Sequential()
        self._init_features(in_shape[1])
        self._init_classifier(out_shape[0])
        self.to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def _init_features(self, in_channels):
        self.features.add_module('conv1', nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1))
        self.features.add_module('bn1', nn.BatchNorm2d(64))
        self.features.add_module('relu1', nn.ReLU(inplace=True))
        self.features.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module('layer1', self._make_layer(64, 128, stride=2))
        self.features.add_module('layer2', self._make_layer(128, 256, stride=2))
        self.features.add_module('layer3', self._make_layer(256, 512, stride=2))
        self.features.add_module('avgpool', nn.AdaptiveAvgPool2d((6, 6)))

    def _make_layer(self, in_channels, out_channels, stride):
        block = nn.Sequential()
        block.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        block.add_module('bn1', nn.BatchNorm2d(out_channels))
        block.add_module('relu1', nn.ReLU(inplace=True))
        block.add_module('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        block.add_module('bn2', nn.BatchNorm2d(out_channels))
        block.add_module('relu2', nn.ReLU(inplace=True))
        return block

    def _init_classifier(self, num_classes):
        self.classifier.add_module('flatten', nn.Flatten())
        self.classifier.add_module('fc1', nn.Linear(512 * 6 * 6, 512))
        self.classifier.add_module('dropout', nn.Dropout(p=0.5))
        self.classifier.add_module('fc2', nn.Linear(512, 256))
        self.classifier.add_module('relu2', nn.ReLU(inplace=True))
        self.classifier.add_module('fc3', nn.Linear(256, num_classes))

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