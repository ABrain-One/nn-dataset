import torch
import torch.nn as nn
import math
from typing import OrderedDict
import numpy as np

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        input_size: Tuple[int, int] = in_shape[2:]
        output_size: int = out_shape[0]
        self.hyperparameters = prm

                                                                                  
        self.features = nn.Sequential(
                        
            nn.Conv2d(in_shape[1], 96, kernel_size=7, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

                     
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

                     
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm['dropout']),

                     
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm['dropout']),

                     
            nn.Conv2d(384, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6))
        )

                    
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm['dropout']),
            nn.Linear(512 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm['dropout']),
            nn.Linear(2048, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

                  
supported_model_params: dict = {
    'lr': 0.01,
    'momentum': 0.95,
    'dropout': 0.4
}

addon_accuracy: 0.9718966035467388