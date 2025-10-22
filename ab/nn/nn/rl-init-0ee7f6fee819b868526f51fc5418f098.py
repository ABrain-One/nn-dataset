import torch
from torch import nn

def supported_hyperparameters():
    return ['lr', 'momentum', 'dropout']


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
        self.learning_rate = prm['lr']
        self.batch_size = prm['batch']
        self.dropout_value = prm['dropout']
        self.momentum = prm['momentum']
        self.transform = prm['transform']
        
        self.criteria = nn.CrossEntropyLoss()
        
        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_value),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.out_shape[0])
        )
        
    def _make_layers(self):
        layers = []
        in_channels = 3
        self._last_channels = 256
        layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(192))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()