import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        reduced_dim = out_channels * 4
        layers = []
        layers += [
            # pw
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=False),
            # dw
            nn.Conv2d(reduced_dim, reduced_dim, kernel_size=3, stride=stride, padding=1, bias=False, groups=reduced_dim),
            nn.BatchNorm2d(reduced_dim),
            # pw-linear
            nn.Conv2d(reduced_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if not self.use_res_connect:
            x = self.conv(x)
        else:
            x = x + self.conv(x)
        return torch.relu(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(in_channels=32, out_channels=16, stride=1),
            nn.AvgPool2d(kernel_size=8)
        )
        self.classifier = nn.Linear(16, num_classes)
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = x.mean(dim=(2, 3))
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