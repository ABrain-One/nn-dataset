import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'weight_decay'}

class AirInitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)

class AirUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if stride != 1 or in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.layers(x)
        return self.relu(x + residual)

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.dropout_rate = prm['dropout']
        self.weight_decay = prm.get('weight_decay', 0.0)

        self.features = nn.Sequential(
            AirInitBlock(in_channels=self.in_channels, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            AirUnit(in_channels=64, out_channels=128, stride=2),
            AirUnit(in_channels=128, out_channels=256, stride=2),
            AirUnit(in_channels=256, out_channels=512, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        optimizer_params = {
            'lr': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
        }
        self.optimizer = SGD(self.parameters(), **optimizer_params)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

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
            self.scheduler.step()