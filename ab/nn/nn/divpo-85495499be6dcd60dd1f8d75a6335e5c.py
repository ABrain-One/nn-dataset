import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm
        in_channels = int(in_shape[1]) if len(in_shape) == 4 else int(in_shape[0])
        num_classes = int(out_shape[0])
        self.stem = ConvBlock(in_channels, 32, 3, 1, 1, activation='relu')
        self.pool = nn.MaxPool2d(2, 2)
        self.block1 = SelfAttentionBlock(32, 96, 3, 1, 1, activation='relu')
        self.block2 = SelfAttentionBlock(96, 16, 3, 1, 1, activation='relu')
        self.classifier = nn.Linear(16 * 8 * 8, num_classes)
        self._initialize_weights()
        self.dropout = nn.Dropout(p=self.prm.get('dropout', 0.0)) if self.prm.get('dropout', 0.0) > 0 else nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = nn.AdaptiveAvgPool2d((8, 8))(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

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
            self.optimizer.step()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride, expand, activation):
        layers = []
        layers.append(
            SELayerBlock(in_channels, out_channels, block=blocks, stride=stride, groups=expand, activation=activation))
        return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1,
                 bias=True, activation=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True) if activation == 'relu6' else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class SelfAttentionBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, block=1, stride=1, groups=1, activation=None):
        super(SelfAttentionBlock, self).__init__()
        self.strides = stride
        self.maxpool = nn.MaxPool2d(stride, stride=block, ceil_mode=True)
        self.avgpool = nn.AvgPool2d(stride, stride=block, ceil_mode=True)
        self.left = self._left_branch(in_channels, out_channels, stride)
        self.right = self._right_branch(in_channels, out_channels, stride, groups, activation)

    def _left_branch(self, in_channels, out_channels, stride):
        if stride == 1:
            left_branch = []
        else:
            left_branch = [nn.MaxPool2d(stride, stride=stride, ceil_mode=True)]
        left_branch = left_branch + [
            ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False,
                      activation=None)]
        return nn.Sequential(*left_branch)

    def _right_branch(self, in_channels, out_channels, stride, groups, activation):
        right_branch = []
        if stride == 1:
            right_branch += [ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1,
                                       bias=False, activation=None)]
        else:
            right_branch += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
        right_branch += [ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups,
                                   bias=False, activation=None)]
        right_branch += [SELayerBlock(out_channels, out_channels, block=1, stride=1, groups=groups, activation=activation)]
        return nn.Sequential(*right_branch)

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return right + left


class SELayerBlock(nn.Module):
    def __init__(self, ch, hidden_dim, block=1, stride=1, groups=1, activation=None):
        super(SELayerBlock, self).__init__()
        self.groups = groups
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, ch, 1),
            nn.Hardsigmoid(True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.se(x).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel attention in SE block"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(True)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention in SE block"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, ch, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Hardsigmoid(True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y