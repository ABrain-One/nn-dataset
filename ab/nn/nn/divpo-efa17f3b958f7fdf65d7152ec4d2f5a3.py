import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = self._make_layer(64, 64, 2)
        self.block3 = self._make_layer(64, 128, 2, True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, downsample=False):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 and downsample else 1
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
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
            weight_decay=prm.get('weight_decay', 0.0001)
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


### Budget: 500,000 parameters.
### Constraints: use standard layers only; no pretrained weights.
### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Case: in_shape = (3, 32, 32).
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].
#### Edge-friendly latency constraint; adhere to in_shape (C x H x W) with the first conv's channels derived from in_shape[0].