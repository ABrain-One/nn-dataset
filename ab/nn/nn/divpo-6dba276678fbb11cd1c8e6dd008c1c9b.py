import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def fuse_conv_bn(self):
        fused_conv = torch.nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
            bias=True,
        )

        # prepare filters
        w_conv = self.conv.weight.view(self.conv.out_channels, -1).clone()
        w_bn = self.bn.weight.clone()

        # prepare spatial bias
        b_conv = torch.zeros(self.conv.weight.size(0), dtype=torch.float32)
        b_bn = self.bn.bias.clone()
        b_bn = torch.tensor(self.bn.running_mean).clone() * w_bn
        scale = torch.tensor(self.bn.running_var).clone() ** 0.5
        w_bn = w_bn / scale
        b_bn = b_bn - b_conv * w_bn
        b_conv.add_(b_bn)

        # apply filters
        w_conv.mul_(w_bn[None])
        b_conv.add_(b_bn)

        # apply spatial bias
        b_conv[self.bn.affine] = b_conv[self.bn.affine] - self.bn.eps * w_conv.sum(1)[self.bn.affine]
        fused_conv.weight.data.copy_(w_conv.view(fused_conv.weight.size()))
        fused_conv.bias.data.copy_(b_conv)

        return fused_conv

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(RepVGGBlock, 32, 32, 3, stride=1, padding=1)
        self.layer2 = self._make_layer(RepVGGBlock, 32, 64, 3, stride=2, padding=1)
        self.layer3 = self._make_layer(RepVGGBlock, 64, 128, 3, stride=2, padding=1)
        self.layer4 = self._make_layer(RepVGGBlock, 128, 256, 3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1, padding=1):
        layers = []
        layers.append(block(in_channels, out_channels, 3, stride, padding))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, 3, 1, padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)

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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()