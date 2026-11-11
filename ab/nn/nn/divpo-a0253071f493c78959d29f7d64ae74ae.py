import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.first_conv = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1, bias=False)
        self.first_bn = nn.BatchNorm2d(self.features)
        self.first_relu = nn.ReLU(inplace=True)
        self.second_conv = self._make_layer(32, 3)
        self.third_conv = self._make_layer(64, 4, stride=2)
        self.fourth_conv = self._make_layer(128, 6, stride=2)
        self.fifth_conv = self._make_layer(256, 9)
        self.sixth_conv = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.classifier = nn.Linear(512, self.num_classes)

    def _make_layer(self, output_features, blocks, stride=1):
        layers = []
        layers += [nn.Conv2d(self.features, output_features, kernel_size=3, stride=stride, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(output_features)]
        layers += [nn.ReLU(inplace=True)]
        self.features = output_features
        for i in range(1, blocks):
            layers += [nn.Conv2d(self.features, output_features, kernel_size=3, stride=1, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(output_features)]
            layers += [nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.second_conv(x)
        x = self.third_conv(x)
        x = self.fourth_conv(x)
        x = self.fifth_conv(x)
        x = self.sixth_conv(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=1e-4)

    def learn(self, train_loader):
        self.train()
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()