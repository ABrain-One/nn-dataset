"""
SPAN: Swift Parameter-free Attention Network
Paper: "SPAN: Swift Parameter-free Attention Network for Efficient Super-Resolution"
Source: NTIRE 2024 Efficient SR Challenge (Winner - Overall Performance & Runtime)
GitHub: https://github.com/hongyuanyu/SPAN

Key Features:
- Parameter-free attention mechanism
- Extremely efficient for mobile deployment
- Winner of NTIRE 2024 challenge
- Parameters: ~600K
- Designed specifically for real-time mobile SR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    """Return supported hyperparameters for Optuna optimization"""
    return {'lr'}


class ParameterFreeAttention(nn.Module):
    """Parameter-Free Spatial Attention Module
    Uses spatial mean to compute an attention map, which is more stable than softmax.
    """
    def __init__(self, dim):
        super().__init__()
        
    def forward(self, x):
        # x: (B, C, H, W)
        # Compute spatial attention using channel-wise mean and sigmoid
        attn = torch.sigmoid(torch.mean(x, dim=1, keepdim=True))
        return x * attn


class SPANBlock(nn.Module):
    """SPAN Building Block
    Optimized for stability and mobile efficiency.
    """
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        
        # Convolution layers
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        
        # Parameter-free attention
        self.attn = ParameterFreeAttention(dim)
        
        # Feed-forward network (FFN)
        hidden_dim = int(dim * ffn_scale)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        
        # Activations
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Convolution path with residual
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        
        # Attention modulation
        out = self.attn(out)
        
        # Add identity (Residual)
        out = out + identity
        
        # FFN path with residual
        out = out + self.ffn(out)
        
        return out


class SPAN(nn.Module):
    """SPAN Architecture"""
    def __init__(self, in_channels=3, out_channels=3, dim=48, n_blocks=12, ffn_scale=2.0, upscale=4):
        super().__init__()
        
        # Input projection
        self.to_feat = nn.Conv2d(in_channels, dim, 3, 1, 1)
        
        # SPAN blocks
        self.blocks = nn.Sequential(*[
            SPANBlock(dim, ffn_scale) for _ in range(n_blocks)
        ])
        
        # Output projection and upsampling
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, out_channels * upscale**2, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )
        
    def forward(self, x):
        x = self.to_feat(x)
        x = self.blocks(x) + x  # Global residual
        x = self.to_img(x)
        return x


class Net(nn.Module):
    """Wrapper for LEMUR/NN Dataset framework"""
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        
        # Extract parameters
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        
        # Get channels from shape
        if len(in_shape) == 4:
            in_channels = in_shape[1]
        else:
            in_channels = 3
            
        if len(out_shape) == 4:
            out_channels = out_shape[1]
        else:
            out_channels = 3
        
        # Calculate upscale factor
        if len(in_shape) >= 3 and len(out_shape) >= 3:
            upscale = out_shape[-1] // in_shape[-1]
        else:
            upscale = 4
        
        # Model configuration (optimized for mobile)
        dim = 48  # Feature dimension
        n_blocks = 12  # Number of SPAN blocks
        ffn_scale = 2.0  # FFN expansion ratio
        
        # Create model
        self.model = SPAN(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            n_blocks=n_blocks,
            ffn_scale=ffn_scale,
            upscale=upscale
        )
        
        # Loss and optimizer
        self.criterion = nn.L1Loss()
        self.lr = prm.get('lr', 1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
    def train_setup(self, prm):
        """Setup for training"""
        self.model.train()
        
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def learn(self, data_roll):
        """Training step"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for lr_img, hr_img in data_roll:
            lr_img = lr_img.to(self.device)
            hr_img = hr_img.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            sr_img = self.model(lr_img)
            
            # Compute loss
            loss = self.criterion(sr_img, hr_img)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
