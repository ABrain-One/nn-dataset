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
        # Fuse Conv and BN for optimized inference
        fused_conv = (self.conv.weight / self.bn.running_var.reshape(-1, 1, 1, 1)).reshape(self.conv.weight.shape) * self.bn.weight.reshape(-1, 1, 1, 1)
        bias = self.conv.bias - (self.bn.running_mean * self.bn.weight / self.bn.running_var) + self.bn.bias
        fused_bn = nn.BatchNorm2d(self.conv.out_channels, affine=False).to(self.conv.weight.device)
        fused_bn.running_mean = torch.zeros_like(self.bn.running_mean)
        fused_bn.running_var = torch.ones_like(self.bn.running_var)
        fused_bn.weight = nn.Parameter(self.bn.weight.clone())
        fused_bn.bias = nn.Parameter(bias.clone())
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels,
                              out_channels=self.conv.out_channels,
                              kernel_size=self.conv.kernel_size,
                              stride=self.conv.stride,
                              padding=self.conv.padding,
                              groups=self.conv.groups,
                              bias=False)
        self.conv.weight = fused_conv
        self.bn = fused_bn
        del self.relu

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 32, stride=1)
        self.block2 = Block(32, 64, stride=2)
        self.block3 = Block(64, 64, stride=1)
        self.block4 = Block(64, 128, stride=2)
        self.block5 = Block(128, 128, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
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
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)

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