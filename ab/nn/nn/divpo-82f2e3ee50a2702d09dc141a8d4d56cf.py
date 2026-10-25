import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # Pointwise
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def fuse_conv_bn(self, conv, bn):
        # Fuse conv and bn into one single convolution layer
        with torch.no_grad():
            # init
            fused_weight = conv.weight
            conv_q = torch.nn.quantized.DeQuantize()
            bn_rm = torch.reshape(bn.running_mean, (1, bn.running_mean.shape[0], 1, 1))
            bn_rv = torch.reshape(bn.running_var, (1, bn.running_var.shape[0], 1, 1))
            bn_eps = bn.eps
            bn_rstd = torch.sqrt(bn_rv + bn_eps)
            bn_gamma = torch.reshape(bn.weight, (1, bn.weight.shape[0], 1, 1))
            bn_beta = torch.reshape(bn.bias, (1, bn.bias.shape[0], 1, 1))

            # fused conv
            fused_weight = (bn_gamma / bn_rstd).view(bn_gamma.shape[0], 1, 1, 1) * fused_weight
            fused_bias = bn_gamma * (bn_beta - bn_rm) / bn_rstd + bn_gamma * bn_beta

            return fused_weight, fused_bias

    def fuse(self):
        fused_weight, fused_bias = self.fuse_conv_bn(self.conv[0], self.conv[1])
        self.conv[0] = torch.nn.quantized.FloatFunctional()
        self.conv[1] = torch.nn.quantized.FloatFunctional()
        self.conv[0].weight = torch.nn.Parameter(fused_weight)
        self.conv[1].weight = torch.nn.Parameter(torch.quantization.FloatToQuantized(fused_bias, torch.min(fused_bias), torch.max(fused_bias), dtype=torch.quint8))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            InvertedResidual(32, 32, 2, 6),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InvertedResidual(64, 64, 1, 6),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            InvertedResidual(128, 128, 1, 6),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        self._set_all_parameters(prm)

    def _set_all_parameters(self, prm):
        for n, m in self.features.named_modules():
            if isinstance(m, InvertedResidual):
                if 'expand_ratio' in prm:
                    if n.endswith('conv'):
                        m.conv[0].weight.data.fill_(1)
                if 'expand_ratio' in prm:
                    if n.endswith('conv.0'):
                        m.fuse(m.conv[0], m.conv[1])

        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

class Block(nn.Module):  # Example Block for reference (you can replace it with your own design)
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))