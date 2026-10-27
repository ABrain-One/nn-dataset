import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        hidden_dim = int(inp * expand_ratio)
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

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def fuse_conv_bn(self):
        C, H, W = self.conv[0].weight.shape
        V = self.conv[1].weight.view(C, -1).t()
        T = self.conv[0].weight.view(C, -1)
        BN_mean = self.conv[1].running_mean
        BN_var_sqrt = torch.sqrt(self.conv[1].running_var + 0.000010000000474974513)
        gamma = self.conv[2].weight.squeeze(0).squeeze(0)
        if self.conv[2].bias is None:
            zeros = torch.zeros_like(gamma)
            ones = torch.ones_like(gamma)
            gamma = torch.where(gamma != 0, gamma, zeros)
            gamma = torch.where(gamma != 1, gamma, ones)
        else:
            gamma = self.conv[2].weight.squeeze(0).squeeze(0)
        conv_weight = torch.mm(T, V).view(C, -1, H, W)
        bias = self.conv[2].bias - BN_mean * gamma / BN_var_sqrt
        BN_std = torch.rsqrt(BN_var_sqrt)
        gamma = gamma * BN_std
        return conv_weight, bias, gamma

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = InvertedResidual(32, 32, stride=1, expand_ratio=4)
        self.block2 = InvertedResidual(32, 32, stride=1, expand_ratio=4)
        self.block3 = InvertedResidual(32, 64, stride=2, expand_ratio=4)
        self.block4 = InvertedResidual(64, 64, stride=1, expand_ratio=4)
        self.block5 = InvertedResidual(64, 64, stride=1, expand_ratio=4)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 6 * 6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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