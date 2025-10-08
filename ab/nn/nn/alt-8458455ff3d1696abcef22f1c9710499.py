import torch
import torch.nn.functional as F
from torch import Tensor, nn

import torch, torch.nn as nn



class FusedMBConv(nn.Module):
    class SqueezeExcitation(nn.Module):
        def __init__(self, n_channels: int, reduction: int = 16):
            super().__init__()
            self.layers = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(n_channels, n_channels // reduction, 1),
                nn.GELU(),
                nn.Conv1d(n_channels // reduction, n_channels, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.layers(x) * x

    def __init__(self, d_model: int, p_dropout: float, n_groups: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 5, 1, 2, groups=n_groups),                              
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            self.SqueezeExcitation(d_model * 2),
            nn.Conv1d(d_model * 2, d_model, 1, groups=n_groups),
            nn.BatchNorm1d(d_model),
            nn.Dropout(p_dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x.transpose(1, 2)).transpose(1, 2)


import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
                                                          
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2, bias=False),                              
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

                                                                 
                                                                                                                                               
                                                                         

                                                                             
                                                        
                                                                                                     

                                                                         
        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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