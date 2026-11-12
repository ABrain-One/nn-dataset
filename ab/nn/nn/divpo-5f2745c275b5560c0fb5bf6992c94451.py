import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.conv1 = nn.Conv2d(self.in_channels, self.features, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.features)
        self.layer1 = self._make_layer(32, 32, 1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Linear(128 * 6 * 6, self.num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layer = []
        layer += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)]
        layer += [nn.BatchNorm2d(out_channels)]
        layer += [nn.ReLU(inplace=True)]
        for i in range(1, blocks):
            layer += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)]
            layer += [nn.BatchNorm2d(out_channels)]
            layer += [nn.ReLU(inplace=True)]
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True)

    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()