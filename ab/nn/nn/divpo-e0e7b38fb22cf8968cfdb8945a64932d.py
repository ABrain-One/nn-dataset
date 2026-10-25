import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        layers.append(
            ConvBlock(
                in_channels,
                in_channels * expand_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                use_bn=True,
                use_relu=True,
            )
        )
        layers.append(
            ConvBlock(
                in_channels * expand_ratio,
                in_channels * expand_ratio,
                kernel_size=3,
                stride=stride,
                padding=1,
                use_bn=True,
                use_relu=False,
            )
        )
        layers.append(
            ConvBlock(
                in_channels * expand_ratio,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                use_bn=True,
                use_relu=False,
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        use_bn=True,
        use_relu=True,
    ):
        super(ConvBlock, self).__init__()

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_bn,
            )
        )

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
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
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layers = nn.Sequential(
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 64, 2, 3),
            InvertedResidual(64, 64, 1, 3),
            InvertedResidual(64, 96, 1, 4),
            InvertedResidual(96, 96, 1, 4),
            InvertedResidual(96, 192, 1, 3),
            InvertedResidual(192, 192, 1, 3),
            InvertedResidual(192, 320, 1, 4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = x.view(x.size(0), -1)
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