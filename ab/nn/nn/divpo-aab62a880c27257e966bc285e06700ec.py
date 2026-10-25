import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build__layer(128, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self._last_layer_name = None
        self._last_dim = None
        self.to(device)

      

    def build_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_blocks - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
            self.scheduler.step()

    def build__layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_blocks - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.02),
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

def check_shape(shape, expected):
    if len(shape) == 4 and len(expected) == 3:
        expected = (1, ) + expected
    assert len(shape) == len(expected), 'Shape mismatch: ' + str(shape) + ' vs ' + str(expected)
    for s, e in zip(shape, expected):
        assert s == e, 'Shape mismatch: ' + str(shape) + ' vs ' + str(expected)

def get_equivalent_kernel_bias(conv, bn):
    bn_weight = bn.weight.detach()
    bn_bias = bn.bias.detach()
    bn_mean = bn.running_mean.detach()
    bn_var = bn.running_var.detach()
    eps = bn.eps
    conv_weight = conv.weight.detach()
    # std = sqrt(eps + inv(count))
    std = (bn_var + eps).sqrt()
    # mean = sum(input_channel) / count
    # output_channel = input_channel * weight[i] + bias
    # bias = mean * weight[i]
    bias = bn_mean * conv_weight.squeeze(0).squeeze(2).squeeze(3).mul(bn_weight).sum(-1)
    # weight = weight[i] * std
    weight = conv_weight.squeeze(1).squeeze(2).squeeze(3).mul(bn_weight).div(std)
    return weight, bias

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ConvBn1d(nn.Module):
    def __init__(self, conv, bn):
        super(ConvBn1d, self).__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ConvBnRelu1d(nn.Module):
    def __init__(self, conv, bn, relu=True):
        super(ConvBnRelu1d, self).__init__()
        self.conv_bn = ConvBn1d(conv, bn)
        self.relu = relu

    def forward(self, x):
        x = self.conv_bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.channels = channels
        self.fcn = nn.Conv2d(self.channels, self.channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = x.mean(dim=(2, 3), keepdim=True)
        attn = self.fcn(attn)
        attn = self.sigmoid(attn)
        return x * attn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()