"""
SAFMN: Spatially-Adaptive Feature Modulation Network
Paper: "Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution"
Source: NTIRE 2023 Efficient SR Challenge (Runner-up)
GitHub: https://github.com/sunny2109/SAFMN

Key Features:
- Spatially-Adaptive Feature Modulation (SAFM) mechanism
- Convolutional Channel Mixer (CCM)
- Very efficient: 3x smaller than IMDN
- Parameters: ~600K for dim=36, n_blocks=8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    """Return supported hyperparameters for Optuna optimization"""
    return {'lr'}


class LayerNorm(nn.Module):
    """Layer Normalization for channels-first format"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class CCM(nn.Module):
    """Convolutional Channel Mixer"""
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


class SAFM(nn.Module):
    """Spatially-Adaptive Feature Modulation"""
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([
            nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) 
            for i in range(self.n_levels)
        ])

        # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU() 

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


class AttBlock(nn.Module):
    """Attention Block with SAFM and CCM"""
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        # Multiscale Block
        self.safm = SAFM(dim) 
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale) 

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


class SAFMN(nn.Module):
    """SAFMN Architecture"""
    def __init__(self, in_channels=3, out_channels=3, dim=36, n_blocks=8, ffn_scale=2.0, upscale=4):
        super().__init__()
        
        self.to_feat = nn.Conv2d(in_channels, dim, 3, 1, 1)
        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, out_channels * upscale**2, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
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
        dim = 36  # Feature dimension
        n_blocks = 8  # Number of attention blocks
        ffn_scale = 2.0  # FFN expansion ratio
        
        # Create model
        self.model = SAFMN(
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
