import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        # Fuse conv, bn, and activation
        fused = nn.Sequential(
            nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                bias=True
            ),
            nn.BatchNorm2d(self.conv.out_channels)
        )

        # Copy weight and bias
        fused[0].weight.data = self.conv.weight.data
        fused[0].bias.data = self.conv.bias.data
        fused[1].weight.data = self.bn.weight.data
        fused[1].bias.data = self.bn.bias.data

        # Use fused version
        self.__class__ = type(self.__class__.__name__, (nn.Sequential,), {
            'forward': lambda self, x: self[0](self[1](x))
        })
        self.__dict__ = fused.__dict__
        del fused

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # handle in_shape in channels-first format
        num_classes = int(out_shape[0])
        self.features = nn.Sequential(
            Block(in_channels, 32, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(32, 64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(6)
        )
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        params = 32 * 3 * 3 * 32 + 32 * 3 * 3 * 64 + 32 * 3 * 3 * 128 + 6 * 6 * 128 + 6 * 6 * 128 + 128 * 36 + num_classes
        assert params <= 500000, "Total parameters must be within the limit (500000)"
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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