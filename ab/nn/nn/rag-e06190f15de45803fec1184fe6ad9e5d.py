# Auto-generated single-file for RepVggDw
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- timm.models.repvit.ConvNorm ----
class ConvNorm(nn.Sequential):
    def __init__(self, in_dim, out_dim, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(in_dim, out_dim, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_dim))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

# ---- RepVggDw (target) ----
class RepVggDw(nn.Module):
    def __init__(self, ed, kernel_size, legacy=False):
        super().__init__()
        self.conv = ConvNorm(ed, ed, kernel_size, 1, (kernel_size - 1) // 2, groups=ed)
        if legacy:
            self.conv1 = ConvNorm(ed, ed, 1, 1, 0, groups=ed)
            # Make torchscript happy.
            self.bn = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
            self.bn = nn.BatchNorm2d(ed)
        self.dim = ed
        self.legacy = legacy

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + x)

    def fuse(self):
        conv = self.conv.fuse()

        if self.legacy:
            conv1 = self.conv1.fuse()
        else:
            conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = nn.functional.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1, 1, 1, 1]
        )

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        if not self.legacy:
            bn = self.bn
            w = bn.weight / (bn.running_var + bn.eps) ** 0.5
            w = conv.weight * w[:, None, None, None]
            b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / (bn.running_var + bn.eps) ** 0.5
            conv.weight.data.copy_(w)
            conv.bias.data.copy_(b)
        return conv

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.rep_vgg_dw = RepVggDw(ed=64, kernel_size=3, legacy=False)
        self.classifier = nn.Linear(64, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.rep_vgg_dw(x)
        x = x.mean(dim=(2, 3))
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
