import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        self.expand_ratio = expand_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.expand_ratio != 1:
            layers.append(
                nn.Conv2d(self.in_channels, self.in_channels * self.expand_ratio, 1, 1, 0, bias=False)
            )
            layers.append(nn.BatchNorm2d(self.in_channels * self.expand_ratio))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(self.in_channels * self.expand_ratio, self.in_channels * self.expand_ratio, 3, stride, 1,
                      groups=self.in_channels * self.expand_ratio, bias=False)
        )
        layers.append(nn.BatchNorm2d(self.in_channels * self.expand_ratio))
        layers.append(nn.ReLU(inplace=True))

        if self.expand_ratio != 1:
            layers.append(
                nn.Conv2d(self.in_channels * self.expand_ratio, self.out_channels, 1, 1, 0, bias=False)
            )
            layers.append(nn.BatchNorm2d(self.out_channels))
        else:
            layers.append(
                nn.Conv2d(self.in_channels, self.out_channels, 3, stride, 1, bias=False)
            )
            layers.append(nn.BatchNorm2d(self.out_channels))

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
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers.append(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU(inplace=True))

        inverted_residual = InvertedResidual(32, 32, 1, 2)
        layers.append(inverted_residual)

        self._last_channels = 32
        return nn.Sequential(*layers)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()