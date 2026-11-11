import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.use_res_conn = self.stride == 1 and in_channels == out_channels

        branch_features = out_channels * 2
        layers = []
        layers.append(nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(branch_features))
        layers.append(nn.ReLU6(inplace=True))
        layers.append(nn.Conv2d(branch_features, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU6(inplace=True))
        self.branch = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_conn:
            return x + self.branch(x)
        else:
            return self.branch(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inverted_residual = InvertedResidual(32, 32, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, 10)

        self.to_device()
        self.set_hyperparameters(prm)

    def to_device(self):
        self.to(self.device)

    def set_hyperparameters(self, prm):
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.inverted_residual(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def learn(self, data_roll):
        self.train()
        for batch in data_roll:
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()