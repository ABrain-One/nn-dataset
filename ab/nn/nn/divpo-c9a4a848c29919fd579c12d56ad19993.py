import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        layers.append(InvertedResidualBlock(
            in_channels, out_channels, stride, expand_ratio))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layer(x)
        else:
            return self.layer(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = round(in_channels * expand_ratio)
        layers = []

        if expand_ratio != 1:
            layers.append(
                ConvBN(in_channels, hidden_dim, 1, 1, 0))

        layers.append(ConvBN(hidden_dim, hidden_dim, 3, stride, 1))
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Block(nn.Module):
    EXPANSION = 2

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2):
        super().__init__()
        self.inverted_residual = InvertedResidual(in_channels, out_channels, stride, expand_ratio)

    def forward(self, x):
        return self.inverted_residual(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            Block(32, 32, expand_ratio=2),
            nn.MaxPool2d(2, 2),
            Block(32, 64, stride=2, expand_ratio=2),
            Block(64, 96, expand_ratio=2),
            Block(96, 160, expand_ratio=2),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Linear(6 * 6 * 160, out_shape[0])
        self.in_channels = 160
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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
            self.optimizer.step()