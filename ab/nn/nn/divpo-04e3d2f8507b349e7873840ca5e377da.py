import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        # Fused weight and bias, to reduce params and keep latency low
        assert self.training is False
        if self.conv.bias is not None:
            return nn.Conv2d(
                in_channels=self.conv.in_channels,
                out_channels=self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
                bias=True
            ).requires_grad_(False).to(self.conv.weight.device)

        # Keep the first BN's scale and bias
        fused = nn.BatchNorm2d(
            num_features=self.conv.out_channels,
            eps=self.bn.eps,
            momentum=self.bn.momentum,
            affine=True,
            track_running_stats=True
        ).requires_grad_(False).to(self.conv.weight.device)

        fused.weight.data[:] = self.bn.weight.data[:] / (torch.sqrt(self.bn.running_var.data[:] + self.bn.eps))
        fused.bias.data[:] = self.bn.bias.data[:] - (fused.weight.data[:] * torch.sqrt(self.bn.running_var.data[:] + self.bn.eps)) * self.bn.running_mean.data[:]

        fused.weight.data[:] *= self.conv.weight.data[:]
        fused.bias.data[:] += self.conv.bias.data[:] if self.conv.bias is not None else 0

        def forward(self, x):
            return self.relu(fused(x))

        self.conv = None
        self.bn = None
        self.relu = None
        self.__class__ = nn.Module
        self.forward = forward

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # handles edge case
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 64, stride=2)
        self.block2 = Block(64, 128, stride=2)
        self.block3 = Block(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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