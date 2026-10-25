import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio != 1:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def fuse_conv_bn(self):
        if self.conv[0].groups != self.conv[0].in_channels:
            raise ValueError("fuse_conv_bn only supports pointwise convolutions")
        fused_conv = nn.Conv2d(
            in_channels=self.conv[0].in_channels,
            out_channels=self.conv[0].out_channels,
            kernel_size=self.conv[0].kernel_size,
            stride=self.conv[0].stride,
            padding=self.conv[0].padding,
            groups=self.conv[0].groups,
            bias=True,
        )
        fused_conv.weight.data = self.conv[0].weight.data
        if self.conv[0].use_bias:
            fused_conv.bias.data = self.conv[0].bias.data
        return fused_conv

    def get_equivalent_kernel_bias(self):
        kernel = self.conv[0].weight.data
        if self.conv[0].use_bias:
            bias = self.conv[0].bias.data
        else:
            bias = torch.zeros_like(self.conv[0].weight.data)
        return kernel, bias

    def convert_to_fully_sharded(self):
        if self.conv[0].groups != self.conv[0].in_channels:
            raise ValueError("fuse_conv_bn only supports pointwise convolutions")
        fused_conv = nn.Conv2d(
            in_channels=self.conv[0].in_channels,
            out_channels=self.conv[0].out_channels,
            kernel_size=self.conv[0].kernel_size,
            stride=self.conv[0].stride,
            padding=self.conv[0].padding,
            groups=self.conv[0].in_channels,
            bias=False,
        )
        fused_conv.weight.data = self.conv[0].weight.data.clone()
        return fused_conv


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.layers = nn.Sequential(
            InvertedResidual(32, 32, 1, 1),
            InvertedResidual(32, 32, 2, 2),
            InvertedResidual(32, 32, 1, 4),
            InvertedResidual(32, 32, 1, 4),
            InvertedResidual(32, 32, 1, 8),
            InvertedResidual(32, 32, 1, 8),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()