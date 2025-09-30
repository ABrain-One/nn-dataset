import torch
import torch.nn as nn
import random

def supported_hyperparameters():
    return {'lr': '0.01', 'momentum': '0.85', 'dropout': '0.6'}

class Net(nn.Module):
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        layers = []
        in_channels = in_shape[1]

                                                                       
        random_int1 = random.randint(3, 5)               
        random_int2 = random.randint(1, 3)           
        random_int = random.randint(2, 8)                                          

        layers += [
            nn.Conv2d(in_channels, 32, kernel_size=random_int1,
                      stride=3, padding=random_int2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = 32

        layers += [
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = 256

        layers += [
            nn.Conv2d(in_channels, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        in_channels = 384

        layers += [
            nn.Conv2d(in_channels, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        in_channels = 384

        layers += [
            nn.Conv2d(in_channels, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = 192
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((random_int, random_int))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm['dropout']),
            nn.Linear(in_channels * random_int * random_int, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm['dropout']),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x