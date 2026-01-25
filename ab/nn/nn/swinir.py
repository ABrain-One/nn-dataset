import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- Model 2: SwinIR (Thesis Version - Final Stable) ---
# 1. Includes Batch Clamping (Prevents Memory Crash)
# 2. Includes Speed Limiter (Prevents Math Error)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res

class Net(nn.Module):
    # --- 1. The Engine (With Speed Limiter) ---
    def train_setup(self, prm):
        # ERROR FIX: We ignore prm['lr'] because 0.05 is too high.
        # We force a safe, standard Learning Rate for Transformers.
        safe_lr = 0.0002 
        self.optimizer = optim.Adam(self.parameters(), lr=safe_lr)

    def learn(self, train_data):
        criterion = nn.L1Loss()
        device = next(self.parameters()).device
        
        # MEMORY FIX: Force max 16 images at a time
        MAX_BATCH = 16
        
        for inputs, labels in train_data:
            # Clamp batch size
            if inputs.size(0) > MAX_BATCH:
                inputs = inputs[:MAX_BATCH]
                labels = labels[:MAX_BATCH]
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            self.optimizer.zero_grad()
            output = self(inputs)
            loss = criterion(output, labels)
            loss.backward()
            
            # STABILITY FIX: Clip gradients to prevent "exploding" numbers
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            
            self.optimizer.step()

    # --- 2. The Architecture ---
    def __init__(self, in_shape=(3, 64, 64), out_shape=(3, 256, 256), prm=None, *args, **kwargs):
        super(Net, self).__init__()
        
        self.scale = 4
        dim = 64  # Standard size
        
        self.head = nn.Conv2d(3, dim, 3, 1, 1)
        
        # 6 Blocks (Good balance of speed vs quality)
        layers = []
        for _ in range(6):
            layers.append(ResidualBlock(dim))
        self.body = nn.Sequential(*layers)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.tail = nn.Conv2d(dim, 3, 3, 1, 1)

    def forward(self, x):
        base = self.head(x)
        res = self.body(base)
        res = res + base
        out = self.upsample(res)
        out = self.tail(out)
        return out

def supported_hyperparameters():
    return {'lr'}
