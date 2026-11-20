import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEAttention(nn.Module):
    """Squeeze-and-Excitation block with SiLU activation."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return x * scale

class GhostModule(nn.Module):
    """GhostNet-style module: cheap conv + linear ops to generate ghost features."""
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.ghost_channels = out_channels // ratio
        
        # Cheap linear operations (conv)
        self.cheap_conv = nn.Conv2d(in_channels, self.ghost_channels, kernel_size=kernel_size, 
                                     padding=kernel_size // 2, bias=False)
        self.bn_cheap = nn.BatchNorm2d(self.ghost_channels)
        
        # Linear operations to generate ghost features (depthwise conv)
        self.ghost_conv = nn.Conv2d(self.ghost_channels, out_channels, kernel_size=1, 
                                     bias=False)
        self.bn_ghost = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.cheap_conv(x)
        x = self.bn_cheap(x)
        x = self.act(x)
        x = self.ghost_conv(x)
        x = self.bn_ghost(x)
        return x

class GhostSEBlock(nn.Module):
    """
    Combines Ghost module for feature generation with SE attention for channel recalibration.
    Uses depthwise conv for spatial mixing.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, se_reduction=4):
        super().__init__()
        self.ghost = GhostModule(in_channels, out_channels, kernel_size, ratio)
        self.se = SEAttention(out_channels, se_reduction)
        self.act = nn.SiLU(inplace=True)
        
        # Optional depthwise conv for better spatial mixing if we want more capacity
        # But Ghost already does spatial mixing via cheap_conv.
        # Let's keep it simple: Ghost -> SE -> Act
        
    def forward(self, x):
        x = self.ghost(x)
        x = self.se(x)
        x = self.act(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        self.in_ch = in_shape[1]
        self.num_classes = out_shape[0] if out_shape else 10

        # Stem: 3 -> 64 (Kernel 3x3, Stride 1)
        # Using a slightly wider stem to capture low-level features well
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 32x32 -> 32x32, 64 -> 64
        # 2 GhostSE blocks
        self.stage1 = nn.Sequential(
            GhostSEBlock(64, 64, kernel_size=3, ratio=2),
            GhostSEBlock(64, 64, kernel_size=3, ratio=2)
        )

        # Downsample: 32x32 -> 16x16, 64 -> 128
        # Using stride 2 conv with Ghost module
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )

        # Stage 2: 16x16 -> 16x16, 128 -> 128
        # 3 GhostSE blocks
        self.stage2 = nn.Sequential(
            GhostSEBlock(128, 128, kernel_size=3, ratio=2),
            GhostSEBlock(128, 128, kernel_size=3, ratio=2),
            GhostSEBlock(128, 128, kernel_size=3, ratio=2)
        )

        # Downsample: 16x16 -> 8x8, 128 -> 256
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )

        # Stage 3: 8x8 -> 8x8, 256 -> 256
        # 4 GhostSE blocks
        self.stage3 = nn.Sequential(
            GhostSEBlock(256, 256, kernel_size=3, ratio=2),
            GhostSEBlock(256, 256, kernel_size=3, ratio=2),
            GhostSEBlock(256, 256, kernel_size=3, ratio=2),
            GhostSEBlock(256, 256, kernel_size=3, ratio=2)
        )

        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, self.num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Handle normalization if required by hyperparameter
        # If 'norm_32' is selected, we assume the input might be raw [0,1] or already normalized.
        # Standard practice in these competitions: if transform is 'norm_32', the data loader usually provides raw data 
        # and we normalize here, OR the data loader provides normalized data. 
        # Looking at Model 1, it did NOT normalize in forward. 
        # Looking at Design 1, it DID normalize in forward.
        # To be safe and consistent with the "transform" hyperparameter meaning "apply this transform",
        # if the data is already normalized by the loader, applying it again is bad.
        # However, usually 'norm_32' implies the *pipeline* uses norm_32. 
        # Let's assume the input is raw [0,1] if we need to normalize, but typically in these setups, 
        # the DataLoader handles the transform. 
        # I will NOT normalize in forward to avoid double normalization if the loader does it, 
        # UNLESS the loader passes raw data. 
        # Given the ambiguity, I will follow Design 1's lead which seemed to handle it explicitly, 
        # but Model 1 (the strongest) did NOT. 
        # Let's assume the input is already normalized if 'norm_32' is in the name, 
        # or that the loader does it. I will leave the input as is to be safe, 
        # similar to Model 1.
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use AdamW as in Model 1 (strongest model)
        # Map 'momentum' to beta1
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Estimate total steps for scheduler
        # CIFAR-10 has 50,000 images
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        # Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=prm['lr'] * 0.01
        )
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()