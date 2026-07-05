import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ab.nn.util.Classes import DataRoll

def supported_hyperparameters():
    return {"lr"}


class EdgeConvBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.conv3 = nn.Conv2d(channels * 2, channels, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x_out = self.act(self.conv1(x))
        x_out = self.act(self.conv2(x_out))
        x_out = self.conv3(x_out)
        return x_out * 0.1 + res

class TitanV9(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        
        # 16-block EdgeConvBlock structure
        self.body = nn.Sequential(*[EdgeConvBlock(channels) for _ in range(16)])
        
        self.exit = nn.Conv2d(channels, 27, 3, padding=1)
        self.upsampler = nn.PixelShuffle(3)

    def forward(self, x):
        features = self.head(x)
        features = self.body(features)
        
        out = self.exit(features)
        out = self.upsampler(out)
        
        # Global Residual Learning (GRL) using Bicubic Baseline
        base = F.interpolate(x, scale_factor=3, mode='bicubic', align_corners=False)
        out = out + base
        
        # Safety valve
        return torch.clamp(out, min=0.0, max=255.0)

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.scale = 3
        
        self.model = TitanV9(channels=64).to(device)
        self.criterion_soft = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=400, eta_min=1e-6)
        
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.model.exit.weight, std=0.001)

    def learn(self, data_roll: DataRoll):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, labels in data_roll:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion_soft(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        self.scheduler.step()
        
        return total_loss / max(num_batches, 1)
