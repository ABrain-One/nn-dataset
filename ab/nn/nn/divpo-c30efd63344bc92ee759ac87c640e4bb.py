import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.features = 32
        self.conv1 = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1)
        self.conv2 = self._make_layer(32, 32, 1)
        self.conv3 = self._make_layer(32, 64, 16, stride=2)
        self.conv4 = self._make_layer(64, 64, 64, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, self.num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layer = []
        downsample = None
        if stride != 1 or self.features != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.features, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layer.append(nn.Conv2d(self.features, out_channels, kernel_size=3, padding=1, stride=stride))
        layer.append(nn.BatchNorm2d(out_channels))
        layer.append(nn.ReLU(inplace=True))
        self.features = out_channels
        for i in range(1, blocks):
            layer.append(nn.Conv2d(self.features, out_channels, kernel_size=3, padding=1))
            layer.append(nn.BatchNorm2d(out_channels))
            layer.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.learning_rate, 
            momentum=self.momentum, 
            weight_decay=5e-4
        )

    def learn(self, train_loader):
        self.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=3.0)
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)