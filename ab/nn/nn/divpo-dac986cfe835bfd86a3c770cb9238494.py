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
        # Fuse conv, bn, and activation
        if self.conv.in_channels == self.conv.out_channels and self.conv.padding == 1:
            fused_conv = nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                groups=self.conv.groups,
                bias=True,
            )
            fused_conv.weight.data = self.conv.weight.data
            fused_conv.bias.data = self.bn.bias.data
            fused_bn = nn.BatchNorm2d(self.conv.out_channels)
            fused_bn.weight.data = self.bn.weight.data
            fused_bn.bias.data = self.bn.bias.data
            fused_bn.running_mean.data = self.bn.running_mean.data
            fused_bn.running_var.data = self.bn.running_var.data
            fused_bn._mean.data = self.bn.running_mean.data
            fused_bn._var.data = self.bn.running_var.data
            return nn.Sequential(fused_conv, fused_bn)
        else:
            return self

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # handle channel-first input shape
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 32, stride=1)
        self.block2 = Block(32, 64, stride=2)
        self.block3 = Block(64, 64, stride=1)
        self.block4 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = None

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)