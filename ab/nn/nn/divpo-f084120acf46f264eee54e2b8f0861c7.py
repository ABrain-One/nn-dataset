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
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)
            )
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            # depthwise
            nn.Conv2d(hidden_dim if expand_ratio != 1 else in_channels,
                      hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pointwise
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
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
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.block1 = InvertedResidual(32, 16, stride=1, expand_ratio=3)
        self.block2 = InvertedResidual(16, 24, stride=2, expand_ratio=3)
        self.block3 = InvertedResidual(24, 24, stride=1, expand_ratio=3)
        self.block4 = InvertedResidual(24, 32, stride=2, expand_ratio=3)
        self.block5 = InvertedResidual(32, 32, stride=1, expand_ratio=3)
        self.block6 = InvertedResidual(32, 64, stride=2, expand_ratio=3)
        self.block7 = InvertedResidual(64, 64, stride=1, expand_ratio=3)

        self.conv = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Linear(16, num_classes)

        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'],
                                         momentum=prm['momentum'],
                                         weight_decay=0.00004)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.conv(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'],
                                         momentum=prm['momentum'],
                                         weight_decay=0.00004)

    def learn(self, train_data):
        self.train()
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()