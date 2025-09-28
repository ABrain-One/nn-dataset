# Auto-generated single-file for LevitMlp
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- timm.models.levit.ConvNorm ----
class ConvNorm(nn.Module):
    def __init__(
            self, in_chs, out_chs, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.linear = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)

        nn.init.constant_(self.bn.weight, bn_weight_init)

    def fuse(self):
        c, bn = self.linear, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1), w.size(0), w.shape[2:], stride=self.linear.stride,
            padding=self.linear.padding, dilation=self.linear.dilation, groups=self.linear.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        return self.bn(self.linear(x))

# ---- timm.models.levit.LinearNorm ----
class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bn_weight_init=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

        nn.init.constant_(self.bn.weight, bn_weight_init)

    def fuse(self):
        l, bn = self.linear, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        x = self.linear(x)
        return self.bn(x.flatten(0, 1)).reshape_as(x)

# ---- LevitMlp (target) ----
class LevitMlp(nn.Module):
    """ MLP for Levit w/ normalization + ability to switch btw conv and linear
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            use_conv=False,
            act_layer=nn.SiLU,
            drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        ln_layer = ConvNorm if use_conv else LinearNorm

        self.ln1 = ln_layer(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.ln2 = ln_layer(hidden_features, out_features, bn_weight_init=0)

    def forward(self, x):
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.ln2(x)
        return x

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
        self.levit_mlp = LevitMlp(
            in_features=32,
            hidden_features=64,
            out_features=32,
            use_conv=True,
            act_layer=nn.SiLU,
            drop=0.0
        )
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.levit_mlp(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()
