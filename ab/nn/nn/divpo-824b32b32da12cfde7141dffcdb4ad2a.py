import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Net class with helper blocks
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        layers.append(
            nn.Conv2d(in_channels,
                      in_channels * expand_ratio,
                      1,
                      stride=1,
                      padding=0,
                      groups=in_channels,
                      bias=False))
        layers.append(nn.BatchNorm2d(in_channels * expand_ratio))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(in_channels * expand_ratio,
                      in_channels * expand_ratio,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=in_channels * expand_ratio,
                      bias=False))
        layers.append(nn.BatchNorm2d(in_channels * expand_ratio))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(in_channels * expand_ratio,
                      out_channels,
                      1,
                      stride=1,
                      padding=0,
                      groups=1,
                      bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            InvertedResidual(in_channels, in_channels, 1, 6),
            InvertedResidual(in_channels, in_channels, 1, 6),
            InvertedResidual(in_channels, in_channels, 1, 6),
        )

    def forward(self, x):
        return x + self._next_channel_same(x, self.block(x))

    def _next_channel_same(self, x, y):
        if x.size(1) == y.size(1):
            return y
        return self._next_channel_diff(x, y)

    def _next_channel_diff(self, x, y):
        c = x.size(1)
        if c < y.size(1):
            return self._next_channel_up(x, y)
        else:
            return self._next_channel_down(x, y)

    def _next_channel_up(self, x, y):
        c = y.size(1)
        return nn.Conv2d(x.size(1), c, 1, 1, bias=False).to(x.device)(x) + y

    def _next_channel_down(self, x, y):
        c = y.size(1)
        return nn.Conv2d(y.size(1), c, 1, 1, bias=False).to(y.device)(y) + x


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        layers = []
        layers += [
            Block(32),
            nn.Sequential(
                InvertedResidual(32, 32, 2, 6),
                InvertedResidual(32, 32, 1, 6)
            )
        ]
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(32 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.avg_pool(x)
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
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()