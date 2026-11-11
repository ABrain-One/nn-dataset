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
        layers += [nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1, bias=False)]
        layers += [nn.BatchNorm2d(in_channels * expand_ratio)]
        layers += [nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3, stride=stride, padding=1, groups=in_channels * expand_ratio, bias=False)]
        layers += [nn.BatchNorm2d(in_channels * expand_ratio)]
        layers += [nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1, bias=False)]
        layers += [nn.BatchNorm2d(out_channels)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            InvertedResidual(32, 32, stride=1, expand_ratio=1),
            nn.MaxPool2d(2, 2),
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            nn.MaxPool2d(2, 2),
            InvertedResidual(64, 128, stride=2, expand_ratio=6),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 1024),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.0)),
            nn.Linear(1024, num_classes)
        )
        self.to(device)

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        lr = prm.get('lr', 1e-2)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()