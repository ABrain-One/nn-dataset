import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict

# --- MANDATORY FOR EVAL ENGINE ---
def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

# --- Building Blocks ---
class DepthwiseSeparableConv(nn.Module):
    """Efficient convolution using depthwise + pointwise."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class GatedConvBlock(nn.Module):
    """
    Combines a standard conv path and a depthwise path, gated by a learned sigmoid.
    This allows the network to dynamically weigh local vs global features.
    """
    def __init__(self, channels, dropout_prob=0.0):
        super().__init__()
        # Path 1: Standard 3x3 conv (captures broader context)
        self.path_std = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        # Path 2: Depthwise 3x3 (captures local texture/edges efficiently)
        self.path_dw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        # Gate: 1x1 conv producing channels
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        x_std = self.path_std(x)
        x_dw = self.path_dw(x)
        g = self.gate(x)
        # Combine: gate * std + (1 - gate) * dw
        out = g * x_std + (1 - g) * x_dw
        out = self.dropout(out)
        # Residual connection
        out = out + x
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm

        c_in = 3
        n_classes = out_shape[0] if out_shape else 10

        # Start with a larger receptive field to reduce spatial dims quickly
        # 32x32 -> 16x16 (stride 2)
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 16x16, 96 channels
        self.stage1 = nn.Sequential(
            GatedConvBlock(96, dropout_prob=0.0),
            GatedConvBlock(96, dropout_prob=0.0),
            nn.AvgPool2d(2) # 16 -> 8
        )

        # Stage 2: 8x8, 192 channels
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            GatedConvBlock(192, dropout_prob=0.0),
            GatedConvBlock(192, dropout_prob=0.0),
            GatedConvBlock(192, dropout_prob=0.0),
            nn.AvgPool2d(2) # 8 -> 4
        )

        # Stage 3: 4x4, 384 channels
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            GatedConvBlock(384, dropout_prob=0.0),
            GatedConvBlock(384, dropout_prob=0.0),
            nn.AvgPool2d(2) # 4 -> 2
        )

        # Final: 2x2, 384 channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(384, n_classes)
        )

        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans2(x)
        x = self.stage2(x)
        x = self.trans3(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        # Use SGD with momentum as per standard CIFAR practice for stability in 1 epoch
        # prm['momentum'] is passed and used
        # prm['lr'] is passed and used
        # prm['batch'] is passed (used by dataloader externally, but we check it exists)
        # prm['transform'] is passed (used by dataloader externally, but we check it exists)
        
        # We use SGD because Adam often struggles to converge well in a single epoch without careful tuning,
        # whereas SGD with momentum is robust.
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Store batch size for potential scheduling if needed, though we do a simple linear decay
        self.batch_size = prm['batch']
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        total_steps = len(train_data)
        
        # Linear decay schedule within the single epoch
        # Start at lr, end at 0
        # We update the learning rate manually each step
        
        for i, (inputs, labels) in enumerate(train_data):
            # Compute current lr
            progress = i / max(total_steps - 1, 1)
            current_lr = self.prm['lr'] * (1 - progress)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()