import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pointwise
            layers.append(
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.append(
            nn.Conv2d(
                hidden_dim if expand_ratio != 1 else in_channels,
                hidden_dim,
                3,
                stride,
                1,
                groups=hidden_dim,
                bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        if expand_ratio != 1:
            # pointwise
            layers.append(
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
        else:
            layers.append(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride, expand_ratio=4),
            InvertedResidual(out_channels, out_channels, 1, expand_ratio=4),
        )

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
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layers = nn.Sequential(
            Block(64, 80, stride=2),
            Block(80, 80, stride=1),
            Block(80, 96, stride=2),
            Block(96, 96, stride=1),
            Block(96, 96, stride=1),
            Block(96, 144, stride=2),
            Block(144, 144, stride=1),
            Block(144, 160, stride=1),
            Block(160, 160, stride=1),
            Block(160, 176, stride=2),
            Block(176, 176, stride=1),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(176, 176, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(176),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Linear(176, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.conv(x)
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