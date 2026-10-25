import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.make_layer(32, 32, 2, 1)
        self.layer2 = self.make_layer(32, 64, 2, 2)
        self.layer3 = self.make_layer(64, 128, 2, 2)
        self.layer4 = self.make_layer(128, 256, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, out_shape[0])
        self._is_training = False

    def make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        for _ in range(blocks - 1):
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def train_setup(self, prm):
        self.to(self.device)
        self._is_training = True
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        if not self._is_training:
            self.train_setup(prm={'lr': 0.01, 'momentum': 0.9})
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}