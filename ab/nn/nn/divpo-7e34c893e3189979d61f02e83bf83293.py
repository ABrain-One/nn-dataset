import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1]) if len(in_shape) == 4 else int(in_shape[0])
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = self._make_layer(32, 32, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.block2 = self._make_layer(32, 64, 2)
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = [nn.BatchNorm2d(in_channels)]
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        for _ in range(1, blocks):
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = nn.AdaptiveAvgPool2d((8, 8))(x)
        x = x.view(x.size(0), -1)
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