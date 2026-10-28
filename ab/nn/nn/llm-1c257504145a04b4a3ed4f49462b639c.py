from typing import Optional
import torch
from torch import nn

def supported_hyperparameters():
    return {'lr', 'momentum', 'stochastic_depth_prob', 'norm_eps', 'norm_std'}

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.num_classes = out_shape[0]
        in_channels = in_shape[1]
        self.stochastic_depth_prob = prm.get('stochastic_depth_prob', 0.1)
        self.norm_eps = prm.get('norm_eps', 1e-06)
        self.norm_std = prm.get('norm_std', 0.02)
        self.channels = (64, 128, 256, 256)
        self.conv1 = nn.Conv2d(in_channels, self.channels[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0], eps=self.norm_eps)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.channels[1], eps=self.norm_eps)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.channels[2], eps=self.norm_eps)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.channels[3], eps=self.norm_eps)
        self.relu4 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=self.stochastic_depth_prob)
        self.fc = nn.Linear(self.channels[3], self.num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()