import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}


def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn(in_channels, hidden_dim, stride))
        layers.extend([
            conv_bn(hidden_dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            conv_bn(32, 32, 1),
            InvertedResidual(32, 32, 1, expand_ratio=3),
            InvertedResidual(32, 32, 1, expand_ratio=3),
            InvertedResidual(32, 32, 1, expand_ratio=3),
            InvertedResidual(32, 32, 1, expand_ratio=3)
        )

        self.layer2 = nn.Sequential(
            conv_bn(32, 64, 2),
            InvertedResidual(64, 64, 1, expand_ratio=3),
            InvertedResidual(64, 64, 1, expand_ratio=3),
            InvertedResidual(64, 64, 1, expand_ratio=3),
            InvertedResidual(64, 64, 1, expand_ratio=3)
        )

        self.layer3 = nn.Sequential(
            conv_bn(64, 128, 2),
            InvertedResidual(128, 128, 1, expand_ratio=3),
            InvertedResidual(128, 128, 1, expand_ratio=3),
            InvertedResidual(128, 128, 1, expand_ratio=3),
            InvertedResidual(128, 128, 1, expand_ratio=3)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()