import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, dropout):
        super().__init__()
        self.stride = stride
        self.dropout = dropout
        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                )
            )
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        if in_channels != out_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, dropout):
        super().__init__()
        self.inverted_residual = InvertedResidual(in_channels, out_channels, stride, expand_ratio, dropout)

    def forward(self, x):
        return self.inverted_residual(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        dropout = prm.get('dropout', 0.0)
        self.dropout = dropout
        self.device = device
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block1 = Block(32, 32, 1, 1, dropout)
        self.block2 = Block(32, 64, 2, 6, dropout)
        self.block3 = Block(64, 96, 1, 4, dropout)
        self.block4 = Block(96, 192, 1, 3, dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for i, (x, y) in enumerate(train_data):
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            out = self(x)
            loss = self.criteria(out, y)
            loss.backward()
            self.optimizer.step()