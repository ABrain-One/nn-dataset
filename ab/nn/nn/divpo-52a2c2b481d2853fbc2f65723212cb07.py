import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        layer_channels = [inp]
        if expand_ratio != 1:
            layer_channels.append(inp * expand_ratio)
        layer_channels += [oup]

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(layer_channels[0], layer_channels[1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(layer_channels[1]),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(layer_channels[1], layer_channels[1], 3, stride, 1, groups=layer_channels[1], bias=False),
            nn.BatchNorm2d(layer_channels[1]),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(layer_channels[1], layer_channels[2], 1, 1, 0, bias=False),
            nn.BatchNorm2d(layer_channels[2]),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Linear(64 * 6 * 6, out_shape[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
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

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        layer_channels = [inp]
        if expand_ratio != 1:
            layer_channels.append(inp * expand_ratio)
        layer_channels += [oup]

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(layer_channels[0], layer_channels[1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(layer_channels[1]),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(layer_channels[1], layer_channels[1], 3, stride, 1, groups=layer_channels[1], bias=False),
            nn.BatchNorm2d(layer_channels[1]),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(layer_channels[1], layer_channels[2], 1, 1, 0, bias=False),
            nn.BatchNorm2d(layer_channels[2]),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)