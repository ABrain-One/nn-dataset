import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion, use_se):
        super(InvertedResidual, self).__init__()
        self.use_se = use_se
        self.same_padding = (stride, 1) if in_channels == out_channels and stride > 1 else (1, 1)

        out_channels_exp = in_channels * expansion

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_exp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels_exp),
            nn.ReLU6(inplace=True),

            nn.Conv2d(out_channels_exp, out_channels_exp, 3, stride, self.same_padding, bias=False),
            nn.BatchNorm2d(out_channels_exp),
            nn.ReLU6(inplace=True),

            nn.Conv2d(out_channels_exp, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.skip_add = stride == 1 and in_channels == out_channels
        self.act = nn.ReLU6(inplace=True)

        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(out_channels, out_channels // 4, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, 1, 1, 0, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        out = self.block(x)
        if self.skip_add:
            out = x + out
        if self.use_se:
            out = out * self.se(out)
        out = self.act(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layers = nn.Sequential(
            InvertedResidual(32, 32, 1, 3, False),
            InvertedResidual(32, 64, 2, 3, False),
            InvertedResidual(64, 64, 1, 3, False),
            InvertedResidual(64, 96, 1, 3, True),
            InvertedResidual(96, 96, 1, 3, True),
            InvertedResidual(96, 160, 2, 3, True),
            InvertedResidual(160, 160, 1, 3, True),
            InvertedResidual(160, 320, 1, 3, True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(320, num_classes),
        )

        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        return x

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
            self.optimizer.step()