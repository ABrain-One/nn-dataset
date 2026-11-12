import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        layer_channels = oup // 2
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, layer_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(layer_channels),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(layer_channels, layer_channels, 3, stride, 1, groups=layer_channels, bias=False),
            nn.BatchNorm2d(layer_channels),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(layer_channels, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.features = self.build_features(in_shape, prm)
        self.classifier = nn.Linear(32, out_shape[0])
        self.to(device)

    def build_features(self, in_shape: tuple, prm: dict) -> nn.Sequential:
        in_channels = in_shape[1]
        layers = []
        layers += [
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        layers += [
            InvertedResidual(32, 32, stride=1, expand_ratio=3),
            InvertedResidual(32, 32, stride=1, expand_ratio=3),
        ]
        self.in_channels = 32
        self.image_size = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
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
            self.optimizer.step()