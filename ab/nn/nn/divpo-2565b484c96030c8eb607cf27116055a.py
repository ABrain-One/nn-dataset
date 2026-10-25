import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

    def fuse(self):
        if isinstance(self.conv, nn.Conv2d) and isinstance(self.batchnorm, nn.BatchNorm2d):
            fused_conv = nn.Conv2d(
                in_channels=self.conv.in_channels,
                out_channels=self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
                bias=True
            )
            fused_conv.weight.data = self.conv.weight.data
            if self.conv.bias is None:
                fused_conv.bias.data = torch.zeros(self.conv.out_channels)
            else:
                fused_conv.bias.data = self.conv.bias.data
            bn_weight = self.batchnorm.weight
            bn_bias = self.batchnorm.bias
            bn_running_mean = self.batchnorm.running_mean
            bn_running_var = self.batchnorm.running_var
            gamma = bn_weight
            if gamma.std() > 0:
                gamma.div_(gamma.std())
            delta = -bn_running_mean * gamma
            self.conv = fused_conv
            self.batchnorm = None
            self.relu = None
            self.conv.weight.data.mul_(gamma.view(1, -1, 1, 1)).add_(delta.view(1, -1, 1, 1))
        else:
            return

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        self.num_classes = int(out_shape[0])
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = nn.Sequential(
            Block(self.in_channels, 32, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(32, 64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, stride=1),
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
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', self.learning_rate),
            momentum=prm.get('momentum', self.momentum),
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