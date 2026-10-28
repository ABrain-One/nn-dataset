import torch
import torch.nn as nn
import numpy as np

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

def transform(norm):
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(*norm)])

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.channels = prm.get('channels', [64, 128, 256, 512])
        self.dropout = prm.get('dropout', 0.2)
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.init_channels = self.channels[0]
        self.features = nn.Sequential(ConvBlock(self.in_channels, self.init_channels, stride=1), nn.MaxPool2d(kernel_size=2, stride=2), ConvBlock(self.init_channels, self.channels[1], stride=1), nn.MaxPool2d(kernel_size=2, stride=2), ConvBlock(self.channels[1], self.channels[2], stride=1), nn.MaxPool2d(kernel_size=2, stride=2), ConvBlock(self.channels[2], self.channels[3], stride=1), nn.MaxPool2d(kernel_size=2, stride=2), nn.AdaptiveAvgPool2d((6, 6)), nn.Dropout(self.dropout))
        self.classifier = nn.Sequential(nn.Linear(self.channels[3] * 6 * 6, 256), nn.ReLU(inplace=True), nn.Dropout(self.dropout), nn.Linear(256, out_shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prm = {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.2, 'channels': [64, 128, 256, 512]}
    net = Net((3, 256, 256), (10,), prm, device)
    net.train_setup(prm)
    return net
if __name__ == '__main__':
    main()