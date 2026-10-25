import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        if hasattr(self.bn, 'weight') and self.bn.weight is not None:
            fused_conv = nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                groups=self.conv.groups,
                bias=True
            ).requires_grad_(False)

            fused_conv.weight.data = (self.conv.weight.data * self.bn.weight.data.reshape(-1, 1, 1, 1)).clone()
            if not torch.equal(self.conv.weight, self.bn.weight.reshape(self.conv.weight.shape)):
                fused_conv.weight.data /= self.bn.running_var.reshape(-1, 1, 1, 1) * self.bn.eps ** 0.5 + self.bn.running_mean.reshape(-1, 1, 1, 1)
            if self.conv.bias is not None:
                bias = self.conv.bias - self.bn.running_mean * self.bn.weight.reshape(-1) / (self.bn.running_var + self.bn.eps)
                fused_conv.bias.data = bias.clone()

            self.conv = fused_conv
            delattr(self, "bn")
        self.__class__ = Block

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else int(in_shape[0])
        num_classes = int(out_shape[0])
        self.conv1 = Block(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = Block(32, 64, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = Block(64, 128, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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