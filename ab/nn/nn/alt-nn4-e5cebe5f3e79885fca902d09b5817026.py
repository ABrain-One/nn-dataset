import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union

def supported_hyperparameters() -> dict:
    return {'lr', 'momentum'}

class Net(nn.Module):

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
        )

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        input_channels = in_shape[1]
        num_classes = out_shape[0]

                                           
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=3, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=0.5)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=0.5)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout2d(p=0.5)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout2d(p=0.5)

                                        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
                                
        self.fc1 = nn.Linear(256 * 6 * 6, 3072)
        self.relu5 = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(3072, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.relu5(x)
        
        x = self.fc2(x)
        return x

addon_accuracy: 0.9508566275924256