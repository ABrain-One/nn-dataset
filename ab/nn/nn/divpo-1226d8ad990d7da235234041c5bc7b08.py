import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive from in_shape
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            Block(in_channels, 64),
            nn.MaxPool2d(2, 2),
            Block(64, 128, stride=2),
            nn.MaxPool2d(2, 2),
            Block(128, 256, stride=2),
            nn.MaxPool2d(2, 2),
            Block(256, 512, stride=2)
        )
        self.layer1 = self._make_layer(Block, 512, 512, 2)
        self.layer2 = self._make_layer(Block, 512, 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(4096, num_classes)
        )

    def _make_layer(self, block, in_planes, out_planes, blocks):
        layers = [block(in_planes, out_planes, stride=1)]
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
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