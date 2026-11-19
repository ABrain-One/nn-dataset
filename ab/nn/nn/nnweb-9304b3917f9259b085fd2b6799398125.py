import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- MANDATORY FOR EVAL ENGINE ---
# The original file defined this function. It must be kept exactly as is.
def supported_hyperparameters():
    return {'lr', 'momentum'}

# --- Helper: DropPath (Stochastic Depth) ---
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  when used in parallel.
    https://arxiv.org/abs/1603.09382
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob > 0 and self.training:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # binarize
            random_tensor.div_(keep_prob)
            return x * random_tensor
        else:
            return x

# --- Helper: Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_path_prob=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.drop_path = nn.Identity()
        if drop_path_prob > 0.0:
            self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.drop_path(out)
        out += self.shortcut(identity)
        
        out = self.act(out)
        return out

# --- Main Network ---
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        
        # Extract input channels from in_shape (H, W, C) or (C, H, W)
        # CIFAR-10 is typically (3, 32, 32) in PyTorch, but let's be robust.
        if len(in_shape) == 3:
            # Assume (C, H, W) as is standard for PyTorch
            c_in = in_shape[0]
        else:
            c_in = 3
            
        n_classes = out_shape[0] if out_shape else 10
        
        # Architecture: ResNet-like with 3 stages
        # Stage 1: 32x32 -> 16x16, 64 channels
        # Stage 2: 16x16 -> 8x8, 128 channels
        # Stage 3: 8x8 -> 4x4, 256 channels
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1, drop_path_prob=0.05),
            ResidualBlock(64, 64, stride=2, drop_path_prob=0.10)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, drop_path_prob=0.15),
            ResidualBlock(128, 128, stride=1, drop_path_prob=0.20)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, drop_path_prob=0.25),
            ResidualBlock(256, 256, stride=1, drop_path_prob=0.30)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, n_classes)
        
        # Initialize weights
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm['momentum'], 0.999), weight_decay=1e-4)
        self.max_batches = prm.get('max_batches', None)
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            if self.max_batches is not None and i >= self.max_batches: break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()