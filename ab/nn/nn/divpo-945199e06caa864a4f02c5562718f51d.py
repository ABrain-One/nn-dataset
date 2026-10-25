import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pointwise
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                )
            )
        layers.extend(
            [
                # depthwise
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dim if expand_ratio != 1 else in_channels,
                        hidden_dim,
                        3,
                        stride,
                        1,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ),
                # pointwise
                nn.Sequential(
                    nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(out_channels),
                ),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = InvertedResidual(32, 32, stride=1, expand_ratio=3)
        self.layer2 = InvertedResidual(32, 32, stride=2, expand_ratio=4)
        self.layer3 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=3),
            InvertedResidual(64, 64, stride=1, expand_ratio=3),
        )
        self.layer4 = InvertedResidual(64, 96, stride=2, expand_ratio=4)
        self.layer5 = nn.Sequential(
            InvertedResidual(96, 96, stride=1, expand_ratio=3),
            InvertedResidual(96, 96, stride=1, expand_ratio=3),
            InvertedResidual(96, 160, stride=2, expand_ratio=4),
        )
        self.layer6 = nn.Sequential(
            InvertedResidual(160, 320, stride=1, expand_ratio=1),
            InvertedResidual(320, 1280, stride=2, expand_ratio=4),
        )
        self.layer7 = InvertedResidual(1280, 1280, stride=1, expand_ratio=1)

        self.features = nn.Sequential(
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
            self.layer7,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(1280, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.to(device)
        self.criteria = (
            nn.CrossEntropyLoss().to(device),
        )
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm["lr"],
            momentum=prm["momentum"],
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get("lr", 0.02),
            momentum=prm.get("momentum", 0.9),
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