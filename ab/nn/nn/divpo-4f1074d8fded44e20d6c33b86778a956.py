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
            ConvBNReLU(
                in_channels,
                in_channels * expand_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        layers.append(
            DepthwiseConvBNReLU(
                in_channels * expand_ratio,
                in_channels * expand_ratio,
                kernel_size=3,
                stride=stride,
                padding=1,
            )
        )
        layers.append(
            ConvBNReLU(
                in_channels * expand_ratio,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                use_activation=False,
            )
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        use_activation=True,
        use_bn=True,
    ):
        super(ConvBNReLU, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
        )
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_activation:
            layers.append(nn.ReLU6(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class DepthwiseConvBNReLU(ConvBNReLU):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        use_activation=True,
    ):
        super(DepthwiseConvBNReLU, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            use_activation=use_activation,
            use_bn=True,
        )

    def conv_bn_layer(self):
        return (
            self.conv[0]
            if len(self.conv) == 1
            else nn.Sequential(self.conv[0], self.conv[1])
        )

    def forward(self, x):
        if self.conv[0].groups == x.shape[1]:
            return super(DepthwiseConvBNReLU, self).forward(x)
        else:
            return self.conv_bn_layer()(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]

        self.features = nn.Sequential(
            ConvBNReLU(in_channels, 32, kernel_size=3, stride=2, padding=1),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            ConvBNReLU(32, 64, kernel_size=3, stride=2, padding=1),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm["lr"],
            momentum=prm["momentum"],
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm["lr"],
            momentum=prm["momentum"],
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