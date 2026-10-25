import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        layers.append(
            nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, stride=1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels * expansion_factor,
                in_channels * expansion_factor,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * expansion_factor,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.layer1 = self._make_layer(32, 32, 3, 1)
        self.layer2 = self._make_layer(32, 32, 3, 1)
        self.layer3 = self._make_layer(32, 32, 3, 2)
        self.layer4 = self._make_layer(32, 32, 3, 1)
        self.layer5 = self._make_layer(32, 32, 3, 1)

        self.conv = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(16 * 6 * 6, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(InvertedResidual(in_channels, out_channels, stride, 3))
        for _ in range(blocks - 1):
            layers.append(InvertedResidual(out_channels, out_channels, 1, 3))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()