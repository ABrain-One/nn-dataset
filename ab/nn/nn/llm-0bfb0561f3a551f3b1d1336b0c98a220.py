import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_prob=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.features = nn.Sequential()
        self.features.add_module('conv1', ConvBlock(in_shape[1], 64, 3, 1, 1, 0.0))
        self.features.add_module('pool1', nn.MaxPool2d(2, 2))
        self.features.add_module('conv2', ConvBlock(64, 128, 3, 1, 1, 0.0))
        self.features.add_module('pool2', nn.MaxPool2d(2, 2))
        self.features.add_module('conv3', ConvBlock(128, 256, 3, 1, 1, 0.0))
        self.features.add_module('pool3', nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential()
        self.classifier.add_module('fc1', nn.Linear(256 * 4 * 4, 512))
        self.classifier.add_module('relu1', nn.ReLU(inplace=True))
        self.classifier.add_module('dropout1', nn.Dropout(0.25))
        self.classifier.add_module('fc2', nn.Linear(512, 512))
        self.classifier.add_module('relu2', nn.ReLU(inplace=True))
        self.classifier.add_module('dropout2', nn.Dropout(0.25))
        self.classifier.add_module('fc3', nn.Linear(512, out_shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x