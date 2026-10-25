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
            # pw
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                )
            )
        layers.extend(
            [
                # dw
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ),
                # pw-linear
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
        )

        self.layers = nn.Sequential(
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 64, 2, 4),
            InvertedResidual(64, 64, 1, 4),
            InvertedResidual(64, 64, 1, 4),
            InvertedResidual(64, 64, 1, 4),
            InvertedResidual(64, 128, 2, 8),
            InvertedResidual(128, 128, 1, 8),
            InvertedResidual(128, 128, 1, 8),
            InvertedResidual(128, 128, 1, 8),
            InvertedResidual(128, 256, 2, 16),
            InvertedResidual(256, 256, 1, 16),
            InvertedResidual(256, 256, 1, 16),
            InvertedResidual(256, 256, 1, 16),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.conv(x)
        x = self.classifier(x)
        return x.flatten(1)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()