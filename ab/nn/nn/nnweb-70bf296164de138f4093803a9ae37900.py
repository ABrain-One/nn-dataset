import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- MANDATORY FOR EVAL ENGINE ---
# Keeping supported_hyperparameters exactly as in the model to improve
def supported_hyperparameters():
    return {'lr', 'momentum'}

# --- Helper Classes ---

class ChannelAttention(nn.Module):
    """Simple SE-like attention to boost feature selection."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Ensure reduction doesn't make channels 0
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    """Spatial attention to focus on relevant regions."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class AttentionBlock(nn.Module):
    """Combines channel and spatial attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class ConvBlock(nn.Module):
    """Standard convolution block with normalization and activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_attention=False, reduction=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.attention = AttentionBlock(out_channels, reduction) if use_attention else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.attention(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block with optional attention."""
    def __init__(self, in_channels, out_channels, stride=1, use_attention=False, reduction=16):
        super().__init__()
        self.block1 = ConvBlock(in_channels, out_channels, stride=stride, use_attention=use_attention, reduction=reduction)
        self.block2 = ConvBlock(out_channels, out_channels, use_attention=use_attention, reduction=reduction)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block1(x)
        out = self.block2(out)
        out += identity
        out = nn.SiLU(inplace=True)(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        
        c_in = 3
        n_classes = out_shape[0] if out_shape else 10
        
        # Reduced base channels to fit parameter budget
        # Previous: 64, 128, 256, 512
        # New: 32, 64, 128, 256
        base_ch = 32
        
        # Entry: Stem
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32x32 -> 32x32
        # Reduced from 2 blocks to 1 block to save params
        self.stage1 = nn.Sequential(
            ResidualBlock(base_ch, base_ch, stride=1, use_attention=True, reduction=8)
        )
        
        # Stage 2: 32x32 -> 16x16
        # Reduced from 3 blocks to 2 blocks
        self.stage2 = nn.Sequential(
            ResidualBlock(base_ch, base_ch * 2, stride=2, use_attention=True, reduction=8),
            ResidualBlock(base_ch * 2, base_ch * 2, stride=1, use_attention=True, reduction=8)
        )
        
        # Stage 3: 16x16 -> 8x8
        # Reduced from 3 blocks to 2 blocks
        self.stage3 = nn.Sequential(
            ResidualBlock(base_ch * 2, base_ch * 4, stride=2, use_attention=True, reduction=8),
            ResidualBlock(base_ch * 4, base_ch * 4, stride=1, use_attention=True, reduction=8)
        )
        
        # Stage 4: 8x8 -> 4x4
        # Reduced from 2 blocks to 1 block
        self.stage4 = nn.Sequential(
            ResidualBlock(base_ch * 4, base_ch * 8, stride=2, use_attention=True, reduction=8)
        )
        
        # Global Pooling and Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(base_ch * 8, n_classes)
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
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