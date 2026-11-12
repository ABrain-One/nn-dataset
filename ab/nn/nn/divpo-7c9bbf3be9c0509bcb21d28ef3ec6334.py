import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()

        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvBnAct(in_channels, in_channels * expand_ratio, kernel_size=1, bias=False)
            )
        layers.extend([
            ConvBnAct(
                in_channels * expand_ratio,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                use_bn=False,
                bias=False
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, use_bn=True, use_act=True, bias=False):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                dilation=dilation,
                bias=bias
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_act:
            layers.append(nn.ReLU6(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(
            ConvBnAct(3, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, stride=1, expand_ratio=3),
            ConvBnAct(32, 64, kernel_size=3, stride=2, padding=1),
            InvertedResidual(64, 64, stride=1, expand_ratio=3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64, out_shape[0])
        self.to(self.device)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
            self.optimizer.step()