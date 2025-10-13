import torch
import torch.nn as nn
import torch.optim as optim

def supported_hyperparameters():
    return {'lr', 'momentum'}


class DPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DPNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(96, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class DPN68(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DPN68, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(
            *[DPNBlock(96, 96) for _ in range(3)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.model = DPN68(in_shape[1], out_shape[0])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()