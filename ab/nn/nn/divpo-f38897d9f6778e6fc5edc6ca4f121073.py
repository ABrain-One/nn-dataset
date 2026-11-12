import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class FastConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.features = 64
        self.conv1 = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.features)
        self.relu = nn.ReLU(inplace=True)
        self.fast_block1 = self._make_fast_block(self.features, 32, 1)
        self.fast_block2 = self._make_fast_block(32, 64, 2)
        self.fast_block3 = self._make_fast_block(64, 128, 2)
        self.fast_block4 = self._make_fast_block(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Linear(256 * 6 * 6, self.num_classes)

    def _make_fast_block(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 1
            if i == 0:
                stride = 2
                self.features = out_channels
            layers += [FastConvBlock(in_channels, out_channels, stride)]
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fast_block1(x)
        x = self.fast_block2(x)
        x = self.fast_block3(x)
        x = self.fast_block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4,
            nesterov=True
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