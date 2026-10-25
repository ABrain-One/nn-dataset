import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        layers.append(nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion_factor, bias=False))
        layers.append(nn.BatchNorm2d(in_channels * expansion_factor))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            InvertedResidual(32, 16, 2, 6),
            InvertedResidual(16, 24, 2, 6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(24, 32, 1, 6),
            InvertedResidual(32, 32, 2, 6),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)
        self.to(device)
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'] if 'lr' in prm else 0.01, momentum=prm['momentum'] if 'momentum' in prm else 0.9)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'] if 'lr' in prm else 0.01, momentum=prm['momentum'] if 'momentum' in prm else 0.9)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()