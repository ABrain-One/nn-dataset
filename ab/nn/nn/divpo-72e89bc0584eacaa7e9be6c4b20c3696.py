import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expansion != 1:
            layers.append(
                nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(in_channels * expansion))
            layers.append(nn.ReLU6(inplace=False))
        layers.append(
            nn.Conv2d(
                in_channels * expansion,
                in_channels * expansion,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * expansion,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(in_channels * expansion))
        layers.append(nn.ReLU6(inplace=False))
        layers.append(
            nn.Conv2d(
                in_channels * expansion,
                out_channels,
                kernel_size=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InvertedResidual(32, 32, stride=1, expansion=1),
        )
        self.features = nn.Sequential(
            InvertedResidual(32, 32, stride=2, expansion=1),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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
        for x, y in train_data:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criteria(self(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()