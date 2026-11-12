import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.features = 64
        self.conv1 = ConvBlock(self.in_channels, self.features, kernel_size=3, stride=1, padding=1)
        self._set_conv_layers(prm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.last_features, self.num_classes)

    def _set_conv_layers(self, prm):
        settings = [
            ('conv2', 64, 1, 1),
            ('conv3', 128, 2, 2),
            ('conv4', 256, 2, 2),
            ('conv5', 512, 2, 2),
        ]
        last_channels = self.features
        for name, channels, stride, pad in settings:
            setattr(self, name, nn.Conv2d(last_channels, channels, kernel_size=3, stride=stride, padding=pad))
            last_channels = channels
        self.last_features = channels

    def forward(self, x):
        x = self.conv1(x)
        for i in range(2, 6):
            x = getattr(self, f'conv{i}')(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()