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
        if isinstance(self.conv, nn.Conv2d) and isinstance(self.bn, nn.BatchNorm2d):
            fused_conv = torch.nn.quantized.FloatFunctional()
            fused_bn = torch.nn.quantized.FloatFunctional()
            weight = self.conv.weight.float()
            if self.conv.bias is None:
                bias = torch.zeros(weight.shape[0], device=weight.device)
            else:
                bias = self.conv.bias.float()
            q_bn_scale = self.bn.running_var.mul(self.bn.eps).rsqrt_()
            q_bn_bias = self.bn.running_mean.sub_(self.bn.running_mean.mul(self.bn.eps).div_(self.bn.running_var.add_(self.bn.eps)))
            q_bn_bias = q_bn_bias.mul_(q_bn_scale).add_(bias)
            q_bn_scale = q_bn_scale.mul_(self.bn.weight)
            fused_weight = fused_conv.mul(weight, q_bn_scale)
            fused_bias = fused_conv.add(bias, q_bn_bias)
            fused_bn_scale = self.bn.weight.mul(q_bn_scale)
            fused_bn_bias = self.bn.weight.mul(q_bn_bias).add_(self.bn.bias)
            self.conv = torch.nn.quantized.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                self.conv.kernel_size,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups,
                dtype=torch.qint8,
                scale=fused_bn_scale.div_(fused_conv.abs().max()).clamp_(1e-5),
                zero_point=torch.zeros([], dtype=torch.qint8)
            )
            self.conv.weight = torch.nn.Parameter(fused_weight)
            self.conv.bias = torch.nn.Parameter(fused_bias)
            self.bn = torch.nn.quantized.DeQuantize()

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # handle channel order according to dataset
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32, stride=1)
        self.block2 = Block(32, 64, stride=2, padding=0)
        self.block3 = Block(64, 64, stride=1)
        self.block4 = Block(64, 128, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()