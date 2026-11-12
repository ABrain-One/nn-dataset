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
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        body = nn.Sequential()
        body.add_module("initial_conv", nn.Conv2d(self.in_channels, 8, kernel_size=3, padding=1, bias=False))
        body.add_module("initial_bn", nn.BatchNorm2d(8))
        body.add_module("initial_re", nn.ReLU(inplace=True))
        body.add_module("conv1", nn.Conv2d(8, 22, kernel_size=3, stride=1, padding=1, bias=False))
        body.add_module("bn1", nn.BatchNorm2d(22))
        body.add_module("re1", nn.ReLU(inplace=True))
        body.add_module("conv2", nn.Conv2d(22, 22, kernel_size=5, stride=1, padding=2, bias=False))
        body.add_module("bn2", nn.BatchNorm2d(22))
        body.add_module("re2", nn.ReLU(inplace=True))
        body.add_module("conv3", nn.Conv2d(22, 48, kernel_size=3, stride=1, padding=1, bias=False))
        body.add_module("bn3", nn.BatchNorm2d(48))
        self.version = 1
        self._last_channels = 48
        return body

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
            self.optimizer.step()