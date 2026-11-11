import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            ConvBlock(self.in_channels, 32, stride=1)
        ]
        layers += [
            ConvBlock(32, 32, stride=1)
        ]
        layers += [
            ConvBlock(32, 64, stride=2)
        ]
        layers += [
            ConvBlock(64, 64, stride=1)
        ]
        layers += [
            ConvBlock(64, 128, stride=2)
        ]
        layers += [
            ConvBlock(128, 128, stride=1)
        ]
        layers += [
            ConvBlock(128, 256, stride=2)
        ]
        layers += [
            ConvBlock(256, 256, stride=1)
        ]
        layers += [
            ConvBlock(256, 512, stride=2)
        ]
        layers += [
            ConvBlock(512, 512, stride=1)
        ]
        self.last_channels = 512
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
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