import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Simplified version of 'MobileNetV2Block' from timm.models.mobilenetv2. InvertedResidual
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            )
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=False))
        layers.extend([
            nn.Conv2d(hidden_dim if expand_ratio != 1 else inp, hidden_dim,
                      kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.conv1 = nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, stride=1, padding=1)

        self.block1 = InvertedResidual(32, 32, 1, 1)
        self.block2 = InvertedResidual(32, 64, 2, 6)
        self.block3 = InvertedResidual(64, 64, 1, 3)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, out_shape[0])

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))
        self.criteria = nn.CrossEntropyLoss().to(self.device)

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