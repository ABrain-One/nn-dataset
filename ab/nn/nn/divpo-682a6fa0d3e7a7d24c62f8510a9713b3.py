import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

    def fuse(self):
        if isinstance(self.conv[1], nn.BatchNorm2d):
            fused_conv = torch.nn.quantized.FloatFunctional()
            fused_bn = torch.nn.quantized.FloatFunctional()
            w_scale, w_zero_point = self.conv[0].weight.q_scale(), self.conv[0].weight.q_zero_point()
            b_scale, b_zero_point = self.conv[1].weight.q_scale(), self.conv[1].weight.q_zero_point()
            fused_weight = fused_bn.mul(self.conv[0].weight, self.conv[1].weight)
            fused_bias = fused_bn.add_scalar(self.conv[1].bias, b_scale * -self.conv[1].running_mean)
            self.conv = nn.Conv2d(
                in_channels=self.conv[0].in_channels,
                out_channels=self.conv[0].out_channels,
                kernel_size=self.conv[0].kernel_size,
                stride=self.conv[0].stride,
                padding=self.conv[0].padding,
                dilation=self.conv[0].dilation,
                groups=self.conv[0].groups,
                bias=True,
                padding_mode=self.conv[0].padding_mode
            )
            self.conv.weight.data = fused_weight
            self.conv.bias.data = fused_bias
            self.conv.qconfig = self.conv[0].qconfig

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 32, stride=1)
        self.block2 = Block(32, 64, stride=2)
        self.block3 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(device),)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
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

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def __repr__(self):
        return f'Net(in_shape={self.in_shape}, out_shape={self.out_shape}, prm={self.prm}, device={self.device})'