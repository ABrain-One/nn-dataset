import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

    def fuse(self):
        # Fuse batchnorm and ReLU for optimization
        if self.conv.in_channels == self.conv.out_channels and self.conv.padding == 1:
            fused_conv = nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                bias=self.conv.bias is None
            )

            if self.conv.bias is None:
                fused_conv.weight.data = self.batchnorm.weight.data * self.conv.weight.data
            else:
                fused_conv.weight.data = self.batchnorm.weight.data * self.conv.weight.data
                fused_conv.bias.data = self.conv.bias.data

            t_bn_cp = self.batchnorm.running_mean
            scale = self.batchnorm.weight.data
            shift = self.batchnorm.bias.data
            fused_bn_bias = t_bn_cp * scale
            fused_bn_weight = scale * scale * self.batchnorm.running_var.data.add_(1).sqrt_()

            if self.conv.bias is not None:
                fused_bn_bias.add_(self.conv.bias * scale)
            else:
                fused_bn_bias.add_(t_bn_cp * scale)

            fused_conv.bias.data = fused_bn_bias
            fused_conv.running_var.data.copy_(fused_bn_weight)
            fused_conv.running_mean.copy_(t_bn_cp)

            return fused_conv

        return self

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive input channels
        num_classes = int(out_shape[0])

        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)

        self.layer1 = Block(32, 32, stride=1)
        self.layer2 = Block(32, 64, stride=2)
        self.layer3 = Block(64, 128, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

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