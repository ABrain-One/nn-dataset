import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pointwise
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True)
                )
            )
        layers.extend([
            # depthwise
            nn.Sequential(
                nn.Conv2d(hidden_dim if expand_ratio != 1 else in_channels,
                          hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ),
            # pointwise
            nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
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
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = InvertedResidual(32, 32, stride=1, expand_ratio=3)
        self.block2 = InvertedResidual(32, 32, stride=1, expand_ratio=3)
        self.block3 = InvertedResidual(32, 64, stride=2, expand_ratio=3)
        self.block4 = InvertedResidual(64, 64, stride=1, expand_ratio=3)
        self.block5 = InvertedResidual(64, 96, stride=1, expand_ratio=3)
        self.block6 = InvertedResidual(96, 96, stride=1, expand_ratio=3)
        self.block7 = InvertedResidual(96, 192, stride=2, expand_ratio=3)
        self.block8 = InvertedResidual(192, 192, stride=1, expand_ratio=3)

        self.aspp = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU6(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU6(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Flatten()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.aspp(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

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