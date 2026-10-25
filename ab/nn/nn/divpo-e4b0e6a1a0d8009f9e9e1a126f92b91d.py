import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            conv_bn(in_channels=in_shape[1], out_channels=32, stride=2),
            conv_bn(in_channels=32, out_channels=64, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(in_channels=64, out_channels=128, blocks=3, stride=2)
        self.layer2 = self._make_layer(in_channels=128, out_channels=256, blocks=4, stride=2)
        self.layer3 = self._make_layer(in_channels=256, out_channels=512, blocks=6, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, out_shape[0])

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [conv_bn(in_channels, out_channels, stride)]
        for _ in range(blocks - 1):
            layers.append(conv_bn(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def learn(self, train_data):
        self.to(self.device)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'] if 'lr' in prm else 0.01,
            momentum=prm['momentum'] if 'momentum' in prm else 0.9,
            weight_decay=0.0001
        )

    def supported_hyperparameters(self):
        return ['lr', 'momentum']