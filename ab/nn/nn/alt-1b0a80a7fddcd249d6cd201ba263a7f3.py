import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
from math import log2, floor

import torch, torch.nn as nn

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


                                                  
                                                       


import torch.nn as nn

def supported_hyperparameters():
    return {'lr','momentum'}

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
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]

                                                                 
                                                                                                                                               
                                                                         

                                                                             
                                                        
                                                                                                     

                                                                         
        self._last_channels = 64
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


class WinAttnAdapter(nn.Module):
    def __init__(self, dim, window_size, num_heads, block_cls, **kwargs):
        super().__init__()
        self.dim = 64                         
        self.window_size = window_size
        self.num_heads = num_heads
        self.block = block_cls(dim=64, num_heads=num_heads, **kwargs)

    def forward(self, x):
                         
        x = x.permute(0, 2, 3, 1)
                                         
        pad_h = (self.window_size[0] - x.shape[1] % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - x.shape[2] % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h))
                            
        x_windows = x.reshape(-1, self.window_size[0], self.window_size[1], self.dim)
                        
        x_windows = self.block(x_windows)
                            
        x = x_windows.reshape(-1, x.shape[1], x.shape[2], self.dim)
               
        x = x[:, :-pad_h, :-pad_w, :]
                              
        x = x.permute(0, 3, 1, 2)
        return x