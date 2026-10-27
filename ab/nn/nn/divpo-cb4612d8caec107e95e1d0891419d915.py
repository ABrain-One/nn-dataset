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
        # Fused Conv-BN-ReLU for better performance
        if self.conv.in_channels == self.conv.out_channels:
            fused_conv = nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                self.conv.kernel_size,
                self.conv.stride,
                self.conv.padding,
                groups=self.conv.in_channels,
                bias=True,
            )

            # Copy weight
            fused_conv.weight.copy_(self.conv.weight)
            # fused_conv.weight.mul_(self.bn.weight.view(1, -1, 1, 1))

            # Copy bias
            bias = self.conv.bias
            if bias is not None:
                tmp_weight = self.conv.weight.view(self.conv.out_channels, -1)
                tmp_bn = self.bn.weight.view(self.bn.out_channels, -1)
                fused_conv.bias.copy_(torch.mm(tmp_bn, bias).view(-1))
            else:
                fused_conv.bias.copy_(self.bn.bias)

            self.__class__ = nn.Conv2d
            self._is_nontrainable = True
            self.register_buffer('weight', fused_conv.weight)
            if fused_conv.bias is None:
                self.register_buffer('bias', None)
            else:
                self.register_buffer('bias', fused_conv.bias)
            self.kernel_size = self.weight.size()[-2:]
            self.stride = self.conv.stride
            self.padding = self.conv.padding
            self.dilation = self.conv.dilation
            self.groups = self.conv.groups
            del self.conv
            del self.bn

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive input channels
        num_classes = out_shape[0]
        self.conv1 = Block(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block2 = Block(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block3 = Block(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

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
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_device(self, device):
        self.device = device
        self.to(device)