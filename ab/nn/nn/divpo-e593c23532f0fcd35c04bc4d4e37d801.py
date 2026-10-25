import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        expand_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6(inplace=True),
            ])

        layers.extend([
            nn.Conv2d(expand_channels, expand_channels, kernel_size=3, stride=stride, padding=1, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False),
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
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.layer1 = nn.Sequential(
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 32, 1, 2),
        )

        self.layer2 = nn.Sequential(
            InvertedResidual(32, 64, 2, 2),
            InvertedResidual(64, 64, 1, 2),
            InvertedResidual(64, 64, 1, 2),
            InvertedResidual(64, 64, 1, 2),
        )

        self.layer3 = nn.Sequential(
            InvertedResidual(64, 96, 2, 2),
            InvertedResidual(96, 96, 1, 2),
            InvertedResidual(96, 96, 1, 2),
            InvertedResidual(96, 96, 1, 2),
            InvertedResidual(96, 96, 1, 2),
            InvertedResidual(96, 96, 1, 2),
        )

        self.layer4 = nn.Sequential(
            InvertedResidual(96, 192, 2, 2),
            InvertedResidual(192, 192, 1, 2),
            InvertedResidual(192, 192, 1, 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(192, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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
            self.optimizer.step()