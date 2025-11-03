import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, out_features=768):
        super().__init__()
        self.body = nn.Sequential(
            # Stem
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 3
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 5
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 6 (projection)
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, out_features)
        )

    def forward(self, x):
        return self.body(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

class Net(nn.Module):
    def __init__(self, in_channels=3, out_features=768, num_classes=10):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, out_features=out_features)
        self.classifier = nn.Linear(out_features, num_classes)
        self.pos_encoder = PositionalEncoding(d_model=out_features)
        self.flatten = nn.Flatten()
        
    def train_setup(self, lr, momentum):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()
        
    def learn(self, x, y, teacher_forcing=True):
        self.train()
        x = x.to(self.encoder.device)
        y = y.to(self.classifier.device)
        
        # Teacher forcing
        if teacher_forcing:
            output = self.forward_with_teacher_forcing(x, y)
        else:
            output = self.forward_without_teacher_forcing(x)
            
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def forward_with_teacher_forcing(self, x, y):
        # Teacher forcing: we assume x is the input and y is the target
        encoded = self.encoder(x)
        encoded = self.pos_encoder(encoded)
        output = self.classifier(encoded)
        return output
        
    def forward_without_teacher_forcing(self, x):
        # Without teacher forcing, we just predict the next token
        encoded = self.encoder(x)
        encoded = self.pos_encoder(encoded)
        output = self.classifier(encoded)
        return output
        
    def forward(self, x, y=None, teacher_forcing=True):
        # Shape asserts
        assert x.dim() == 4, f"Input must have 4 dimensions, got {x.dim()}"
        assert y is None or y.dim() == 2, f"Target must be None or have 2 dimensions, got {y.dim() if y is not None else 'None'}"
        
        # Teacher forcing
        if teacher_forcing:
            output = self.forward_with_teacher_forcing(x, y)
        else:
            output = self.forward_without_teacher_forcing(x)
            
        return output

# --- auto-closed by AlterCaptionNN ---