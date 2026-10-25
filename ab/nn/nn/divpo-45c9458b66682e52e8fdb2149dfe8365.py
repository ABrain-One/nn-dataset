import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.features = self._make_layers([32, 64, 128, 256])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, self.num_classes)

    def _make_layers(self, out_channels):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(out_channels[2], out_channels[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

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
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=prm.get('weight_decay', 0.0)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}