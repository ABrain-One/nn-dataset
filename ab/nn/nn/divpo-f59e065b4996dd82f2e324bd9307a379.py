import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.layer1 = self._make_layer(32, 32, 3)
        self.layer2 = self._make_layer(32, 64, 3, stride=2)
        self.layer3 = self._make_layer(64, 128, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(128 * 7 * 7, self.num_classes)
        self.apply(self._init_weights)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layer = []
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False))
        layer.append(nn.BatchNorm2d(out_channels))
        layer.append(nn.ReLU(inplace=True))
        for i in range(1, blocks):
            layer.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layer.append(nn.BatchNorm2d(out_channels))
            layer.append(nn.ReLU(inplace=True))
        self.add_module(f"layer{len(layer)}", nn.Sequential(*layer))
        return nn.Sequential(*layer)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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