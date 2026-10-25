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
        # Fuse conv, bn, and relu for better efficiency
        if self.conv.in_channels == self.conv.out_channels:
            fused_conv = (self.conv.weight, self.bn.weight / (torch.sqrt(self.bn.running_var + self.bn.eps) + self.bn.eps),
                          self.bn.bias - self.bn.running_mean * self.bn.weight / (torch.sqrt(self.bn.running_var + self.bn.eps) + self.bn.eps),
                          self.conv.bias if self.conv.bias is None else self.conv.bias - self.bn.running_bias * self.bn.weight / (torch.sqrt(self.bn.running_var + self.bn.eps) + self.bn.eps))
            self.conv = torch.nn.utils.fuse_conv_bn_eval(self.conv, self.bn)
            self.bn = None
            self.__delattr__('bn')
            self.__delattr__('relu')
            self.conv = torch.nn.Conv2d(*fused_conv)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive from in_shape
        num_classes = int(out_shape[0])

        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)

        self.layer1 = Block(32, 32, stride=1)
        self.layer2 = Block(32, 64, stride=2)
        self.layer3 = Block(64, 128, stride=2)
        self.layer4 = Block(128, 128, stride=1)
        self.layer5 = Block(128, 256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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

    def __repr__(self):
        return f'Net(in_shape={self.in_shape}, out_shape={self.out_shape}, prm={self.prm}, device={self.device})'