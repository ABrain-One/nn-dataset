import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.rnn = nn.RNN(10, 20, 2, batch_first=True)
        self.fc = nn.Linear(20, 10)
        
    def train_setup(self, **kwargs):
        pass
        
    def learn(self, **kwargs):
        pass
        
    def forward(self, x, teacher_forcing=False, **kwargs):
        assert x.dim() == 3, "Input must be 3D"
        h = self.rnn(x)[0]
        out = self.fc(h.transpose(0, 1)).transpose(0, 1)
        return out