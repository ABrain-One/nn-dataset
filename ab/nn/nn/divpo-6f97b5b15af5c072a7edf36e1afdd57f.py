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
        # Fused block for edge-friendly deployment
        if not self.training:
            fused_conv = torch.nn.Conv2d(
                in_channels=self.conv.in_channels,
                out_channels=self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                groups=self.conv.groups,
                bias=True
            )
            fused_conv.weight.data = self.conv.weight.data
            if self.conv.bias is not None:
                fused_conv.bias.data = self.conv.bias.data
            bn_weight = self.bn.weight
            bn_bias = self.bn.bias
            bn_mean = self.bn.running_mean
            bn_var_sqrt = torch.sqrt(self.bn.running_var + self.bn.eps)
            fused_conv.bias.data = (bn_bias - bn_mean) / bn_var_sqrt
            fused_conv.weight.data *= (bn_weight / bn_var_sqrt).view(
                fused_conv.out_channels, 1, 1, 1)
            return fused_conv
        return self

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = Block(32, 32, stride=1)
        self.layer2 = Block(32, 64, stride=2)
        self.layer3 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = None

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def fuse(self):
        # Fuse all blocks for edge-friendly deployment
        self.stem = self.stem.fuse()
        self.layer1 = self.layer1.fuse()
        self.layer2 = self.layer2.fuse()
        self.layer3 = self.layer3.fuse()