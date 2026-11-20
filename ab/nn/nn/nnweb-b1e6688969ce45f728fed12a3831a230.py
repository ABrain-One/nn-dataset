import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        out_channels = out_shape[0]

        # Stage 1: 3x3 conv, BN, ReLU -> 64 channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # Stage 2: 3x3 conv stride 2, BN, ReLU -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        # Residual block in deeper feature stage (128 -> 128)
        self.res_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.res_bn1 = nn.BatchNorm2d(128)
        self.res_relu1 = nn.ReLU()
        self.res_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.res_bn2 = nn.BatchNorm2d(128)
        self.res_relu2 = nn.ReLU()

        # Stage 3: 3x3 conv stride 2, BN, ReLU -> 256 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        # Stage 4: 3x3 conv, BN, ReLU -> 256 channels (final conv stage)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        # Classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, out_channels)

    def forward(self, x):
        # Stage 1
        x = self.relu1(self.bn1(self.conv1(x)))

        # Stage 2
        x = self.relu2(self.bn2(self.conv2(x)))

        # Residual block
        identity = x
        out = self.res_relu1(self.res_bn1(self.res_conv1(x)))
        out = self.res_relu2(self.res_bn2(self.res_conv2(out)))
        # Use torch.add instead of in-place += to preserve autograd graph
        x = torch.add(out, identity)

        # Stage 3
        x = self.relu3(self.bn3(self.conv3(x)))

        # Stage 4
        x = self.relu4(self.bn4(self.conv4(x)))

        # Classifier
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x