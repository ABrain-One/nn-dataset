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
        # Fuse conv, bn, and activation for faster inference
        if self.conv.in_channels == self.conv.out_channels:
            fused_conv = nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                groups=self.conv.groups,
                bias=True
            ).requires_grad_(False)

            fused_conv.weight.data = self.conv.weight.data
            if self.bn.weight is not None:
                bn_mean = self.bn.running_mean
                bn_variance = self.bn.running_var
                fused_conv.weight.data *= torch.sqrt(bn_variance + 1e-5) * (bn_mean / torch.sqrt(bn_variance + 1e-5)).view(-1, 1, 1, 1)
                if self.bn.bias is not None:
                    fused_conv.bias.data = self.bn.bias.data - bn_mean * (self.bn.weight.data / torch.sqrt(bn_variance + 1e-5))
            self.fused_kernel = fused_conv
            self.__delattr__('conv')
            self.__delattr__('bn')
            self.conv = self.fused_kernel
            del self.relu

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 64, stride=2)
        self.block2 = Block(64, 128, stride=2)
        self.block3 = Block(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)
        self._set_training_mode(prm)

    def _set_training_mode(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        params = self.parameters()
        self.optimizer = torch.optim.SGD(
            params,
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self._set_training_mode(prm)

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
        # Fuse all blocks for faster inference
        for block in [self.block1, self.block2, self.block3]:
            block.fuse()
        self.stem = self.stem.eval()
        self.classifier = self.classifier.eval()