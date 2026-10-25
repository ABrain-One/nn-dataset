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
        # Fuse conv and bn for inference speedup
        if not self.is_fused:
            fused_conv = nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                self.conv.kernel_size,
                self.conv.stride,
                self.conv.padding,
                bias=True,
            ).requires_grad_(False)
            fused_conv.weight.data = (
                self.conv.weight.data * self.bn.weight.reshape(-1, 1, 1, 1))
            if not torch.equal(self.bn.weight, torch.ones_like(self.bn.weight)):
                fused_conv.bias.data = (
                    self.conv.bias.data +
                    self.bn.bias.data -
                    self.bn.running_mean * self.bn.weight.reshape(-1) /
                    self.bn.running_var.reshape(-1).sqrt())
            self.__class__ = nn.Conv2d
            self.in_channels = fused_conv.in_channels
            self.out_channels = fused_conv.out_channels
            self.kernel_size = fused_conv.kernel_size
            self.stride = fused_conv.stride
            self.padding = fused_conv.padding
            self.weight = fused_conv.weight
            if fused_conv.bias is None:
                self.register_parameter('bias', None)
            else:
                self.bias = fused_conv.bias
            self.is_fused = True

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.is_fused = False
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # handle in_shape as (batch, channels, height, width)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            Block(in_channels, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = Block(64, 128, 3, 1, 1)
        self.layer2 = nn.Sequential(
            Block(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = Block(128, 256, 3, 1, 1)
        self.layer4 = nn.Sequential(
            Block(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def fuse(self):
        for block in self.modules():
            if isinstance(block, Block) and not isinstance(block, nn.Conv2d):
                block.fuse()
                self.is_fused = True