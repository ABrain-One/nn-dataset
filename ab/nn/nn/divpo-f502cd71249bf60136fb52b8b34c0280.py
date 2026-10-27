import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        layers.append(
            ConvBNReLU(
                in_channels,
                in_channels * expand_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU6,
            )
        )

        layers.append(
            ConvBNReLU(
                in_channels * expand_ratio,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU6,
            )
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def fuse_conv_bn(self, conv, bn):
        # Fuse conv and bn
        w = conv.weight.clone()
        if conv.bias is None:
            b = torch.zeros_like(bn.bias)
        else:
            b = conv.bias.clone()
        if bn.eps is None:
            eps = 1e-5
        else:
            eps = bn.eps
        w *= (bn.weight / torch.sqrt(bn.running_var + eps)).reshape(-1, 1, 1, 1)
        b *= bn.weight / torch.sqrt(bn.running_var + eps)
        b += bn.bias - bn.running_mean * w.flatten(1).mean(1)
        return w, b

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU6,
        **kwargs
    ):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
                **kwargs
            ),
            norm_layer(out_channels),
            activation_layer(inplace=True),
        )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.layers = nn.Sequential(
            InvertedResidual(32, 32, stride=1, expand_ratio=3),
            InvertedResidual(32, 32, stride=2, expand_ratio=3),
            InvertedResidual(32, 32, stride=1, expand_ratio=3),
            InvertedResidual(32, 64, stride=2, expand_ratio=3),
            InvertedResidual(64, 64, stride=1, expand_ratio=3),
            InvertedResidual(64, 96, stride=1, expand_ratio=3),
            InvertedResidual(96, 96, stride=1, expand_ratio=3),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=2, expand_ratio=6),
            InvertedResidual(96, 192, stride=1, expand_ratio=6),
        )

        self.conv = ConvBNReLU(192, 192, kernel_size=1, stride=1, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(192, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.conv(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()