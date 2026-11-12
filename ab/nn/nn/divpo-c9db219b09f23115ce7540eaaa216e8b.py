import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, input_channels, output_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = (stride == 1 and input_channels == output_channels)
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(input_channels, input_channels * expansion_factor, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(input_channels * expansion_factor),
                nn.ReLU6(inplace=True),
            )
        )
        layers.append(
            nn.Sequential(
                nn.Conv2d(input_channels * expansion_factor, input_channels * expansion_factor, kernel_size=3, stride=stride, padding=1, groups=input_channels * expansion_factor, bias=False),
                nn.BatchNorm2d(input_channels * expansion_factor),
                nn.ReLU6(inplace=True),
            )
        )
        layers.append(
            nn.Sequential(
                nn.Conv2d(input_channels * expansion_factor, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layer(x)
        else:
            return self.layer(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        layers = []
        layers += [
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        layers += [
            InvertedResidual(32, 32, 2, 3),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)
        self.to(self.device)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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