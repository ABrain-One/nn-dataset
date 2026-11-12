import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = int(in_shape[1]) if len(in_shape) == 4 else int(in_shape[0])
        self.num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.layers = self._make_layers(32, 32, 1, 1)
        self.avgpool = nn.AvgPool2d(2)
        self.classifier = nn.Linear(32, self.num_classes)
        self.criteria = nn.CrossEntropyLoss().to(self.device)

    def _make_layers(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = nn.functional.adaptive_avg_pool2d(x, (3, 3))
        x = self.classifier(x.view(x.size(0), x.size(1), -1).mean(dim=2))
        return x

    def train_setup(self, prm):
        self.to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()