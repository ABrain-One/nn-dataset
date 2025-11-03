

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def supported_hyperparameters():
    return {'lr','momentum'}



class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, in_channels, ratio=1/16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // ratio)
        self.fc2 = nn.Linear(in_channels // ratio, in_channels)
        
    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, output_size=1).reshape(b, c)
        z = torch.sigmoid(self.fc2(F.relu(self.fc1(y))))
        out = x * z.reshape(b, c, 1, 1)
        return out


class PositionalEncoding(nn.Module):
    """Learnable Positional Encoding"""
    def __init__(self, d_model: int, dropout: float, max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
       