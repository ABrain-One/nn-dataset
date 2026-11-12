import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, kernel_size=1, stride=1, padding=0, groups=inp, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, padding=1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.features = self._make_layers([
            [32, 16, 1, 1],
            [16, 24, 2, 2],
            [24, 32, 1, 1],
            [32, 64, 2, 2],
            [64, 96, 1, 1],
            [96, 160, 2, 2],
            [160, 320, 1, 1],
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(320, num_classes)
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))

    def _make_layers(self, block_config):
        layers = []
        for in_channels, out_channels, stride, expand_ratio in block_config:
            layers.append(InvertedResidual(in_channels, out_channels, stride, expand_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'] if 'lr' in prm else 0.01,
            momentum=prm['momentum'] if 'momentum' in prm else 0.9,
        )

    def learn(self, data: iter):
        self.train()
        for batch, labels in data:
            batch, labels = batch.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self(batch)
            loss = self.criteria(output, labels)
            loss.backward()
            self.optimizer.step()