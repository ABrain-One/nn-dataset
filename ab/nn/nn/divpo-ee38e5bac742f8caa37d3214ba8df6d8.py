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
        # Fuse conv, batch norm, and activation
        if hasattr(self.bn, 'eps'):
            bn_w = self.bn.weight / (self.bn.weight.pow(2) + self.bn.eps).sqrt()
            bn_b = self.bn.bias - self.bn.running_mean * bn_w * self.bn.running_var
            conv_w = self.conv.weight * torch.reshape(bn_w, [self.conv.weight.shape[0], 1, 1, 1])
            new_conv = torch.nn.Conv2d(
                in_channels=self.conv.in_channels,
                out_channels=self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                groups=self.conv.groups,
                bias=True
            ).requires_grad_(False)
            new_conv.weight.data = conv_w
            new_conv.bias.data = bn_b
            self.__class__ = torch.nn.Conv2d
            self.__dict__ = new_conv.__dict__
            del self.bn, self.relu

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = int(out_shape[0])
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = nn.Sequential(
            Block(self.in_channels, 32, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(32, 64, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', self.learning_rate),
            momentum=prm.get('momentum', self.momentum),
        )
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()