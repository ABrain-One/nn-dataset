import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion, device):
        super(InvertedResidual, self).__init__()
        self.device = device
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        layers += [
            nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU6(inplace=True),
        ]

        if stride == 1 and in_channels == out_channels:
            layers += [
                nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion, bias=False),
                nn.BatchNorm2d(in_channels * expansion),
                nn.ReLU6(inplace=True),
            ]
        else:
            layers += [
                nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers.append(nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU(inplace=True))

        layers.append(InvertedResidual(32, 32, stride=1, expansion=1, device=self.device))
        layers.append(InvertedResidual(32, 32, stride=1, expansion=1, device=self.device))
        layers.append(InvertedResidual(32, 32, stride=1, expansion=1, device=self.device))
        layers.append(InvertedResidual(32, 32, stride=1, expansion=1, device=self.device))
        layers.append(InvertedResidual(32, 32, stride=1, expansion=1, device=self.device))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

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