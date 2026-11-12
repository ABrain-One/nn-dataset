import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.use_res = stride == 1 and in_channels == out_channels
        exp_channels = in_channels * 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(exp_channels, exp_channels, kernel_size=3, stride=stride, padding=1, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(exp_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.inverted_residual = InvertedResidual(in_channels, out_channels, stride)

    def forward(self, x):
        return self.inverted_residual(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        layers = []
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        layers += [Block(32, 32, 1)]
        layers += [Block(32, 32, 2)]
        layers += [Block(32, 32, 1)]
        layers += [Block(32, 32, 1)]
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)
        self.to_device()
        self.set_prm(prm)

    def to_device(self):
        self.to(self.device)

    def set_prm(self, prm):
        self.prm = prm

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to_device()
        self.set_prm(prm)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, data: list):
        self.train()
        for epoch in range(self.prm['epoch']):
            for x, y in data:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self(x)
                loss = self.criteria[0](out, y)
                loss.backward()
                self.optimizer.step()