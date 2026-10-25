import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels * self.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels * self.expansion, self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

        self.skip = nn.Sequential()
        if stride != 1 or self.in_channels != self.out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.bottleneck(x) + self.skip(x))


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
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        layers += [
            Block(32, 32, 1),
            Block(32, 32, 2),
            Block(32, 32, 1),
            Block(32, 32, 2),
            Block(32, 32, 1),
            Block(32, 32, 2),
            Block(32, 32, 1),
            Block(32, 32, 2),
            Block(32, 32, 1),
            Block(32, 32, 2),
        ]
        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
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