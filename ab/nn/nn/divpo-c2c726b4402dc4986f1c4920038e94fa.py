import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.build_layer(32, 32)
        self.layer2 = self.build_layer(32, 64)
        self.layer3 = self.build_layer(64, 128)
        self.layer4 = self.build_layer(128, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(256 * 7 * 7, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def build_features(self):
        return nn.Sequential()

    def build_layer(self, in_c, out_c):
        return self._make_layer(in_c, out_c, blocks=2)

    def _make_layer(self, in_c, out_c, blocks):
        layer = []
        downsample = nn.Sequential()
        stride, pad = 1, 0
        if in_c != out_c:
            stride, pad = 2, 1
        layer += [
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=pad),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        for i in range(1, blocks):
            layer += self._build_inner_layer(out_c)
        return nn.Sequential(*layer)

    def _build_inner_layer(self, c):
        return [
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        ]

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
            self.optimizer.step()