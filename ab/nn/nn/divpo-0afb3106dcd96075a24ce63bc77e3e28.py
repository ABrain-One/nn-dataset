import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        layers += [
            nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True)
        ]

        layers += [
            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True)
        ]

        layers += [
            nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.url = True
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            InvertedResidual(32, 16, 1, 3),
            InvertedResidual(16, 24, 2, 3),
            InvertedResidual(24, 32, 1, 3),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            B = x.shape[0]
            x = x.unsqueeze(dim=1)
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        x = x.flatten(1)
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