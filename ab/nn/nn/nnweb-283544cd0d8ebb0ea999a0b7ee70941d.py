import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- MANDATORY FOR EVAL ENGINE ---
def supported_hyperparameters():
    return {'lr', 'momentum'}

# --- Helper Classes ---

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class EfficientBlock(nn.Module):
    """
    A lightweight block based on depthwise separable convolutions.
    Structure: BN -> GELU -> Depthwise Conv 3x3 -> BN -> GELU -> Pointwise Conv 1x1 -> BN
    """
    def __init__(self, in_channels, out_channels, expansion=2):
        super(EfficientBlock, self).__init__()
        mid_channels = in_channels * expansion
        
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            GELU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        
        # Derive input channels
        if len(in_shape) == 3:
            c_in = in_shape[0]
        else:
            c_in = 3
            
        n_classes = out_shape[0] if out_shape else 10

        # Architecture Design:
        # 1. Stem: Conv 3x3 -> BN -> GELU
        # 2. Stage 1: 2 EfficientBlocks (channels: 32 -> 64)
        # 3. Stage 2: 3 EfficientBlocks (channels: 64 -> 128)
        # 4. Stage 3: 4 EfficientBlocks (channels: 128 -> 256)
        # 5. Head: Global Average Pooling -> Dropout -> Linear
        
        # Parameter budget check:
        # Stem: 3*32*3*3 + 32 = 864
        # Stage 1 (32->64): 
        #   Block 1 (32->32): 32*32*1 + 32*32*9 + 32*32*1 = 1024 + 8192 + 1024 = 10240
        #   Block 2 (32->64): 32*64*1 + 64*64*9 + 64*64*1 = 2048 + 36864 + 4096 = 43008
        # Stage 2 (64->128):
        #   Block 3 (64->64): 64*64*1 + 64*64*9 + 64*64*1 = 4096 + 36864 + 4096 = 45056
        #   Block 4 (64->128): 64*128*1 + 128*128*9 + 128*128*1 = 8192 + 147456 + 16384 = 172032
        #   Block 5 (128->128): 128*128*1 + 128*128*9 + 128*128*1 = 16384 + 147456 + 16384 = 180224
        # Stage 3 (128->256):
        #   Block 6 (128->128): 180224
        #   Block 7 (128->256): 128*256*1 + 256*256*9 + 256*256*1 = 32768 + 589824 + 65536 = 688128
        #   Block 8 (256->256): 256*256*1 + 256*256*9 + 256*256*1 = 65536 + 589824 + 65536 = 720896
        #   Block 9 (256->256): 720896
        # Head: 256 * 10 = 2560
        # Total approx: ~1.7M params. Well within 5M limit.

        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            GELU()
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            EfficientBlock(32, 32),
            EfficientBlock(32, 64)
        )
        
        # Pooling 1
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Stage 2
        self.stage2 = nn.Sequential(
            EfficientBlock(64, 64),
            EfficientBlock(64, 128),
            EfficientBlock(128, 128)
        )
        
        # Pooling 2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Stage 3
        self.stage3 = nn.Sequential(
            EfficientBlock(128, 128),
            EfficientBlock(128, 256),
            EfficientBlock(256, 256),
            EfficientBlock(256, 256)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )

        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Model 1 (74.66%) used AdamW with lr=0.002, momentum=0.75
        # We use SGD with momentum because it is often more stable for 1-epoch training
        # when combined with a proper LR schedule.
        # Let's stick to the strong signal from Model 1: AdamW.
        # But Model 1's success might be due to the specific architecture + AdamW.
        # Let's try SGD with a cosine schedule, which is the gold standard for 1-epoch CIFAR.
        # Actually, let's look at the hyperparameters provided.
        # If I use SGD, I should use a high LR.
        # If I use AdamW, I should use a low LR.
        
        # Let's use SGD with momentum as it's robust.
        # But wait, Model 1 used AdamW.
        # Let's try a hybrid: AdamW with a cosine schedule.
        
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Store batch size
        self.batch_size = prm.get('batch', 128)
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
        
        # Cosine schedule within 1 epoch
        # Start at lr, end at 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=0.0
        )
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            
            self.optimizer.step()
            scheduler.step()