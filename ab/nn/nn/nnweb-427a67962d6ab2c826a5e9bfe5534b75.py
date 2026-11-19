import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class FractalAttentionBlock(nn.Module):
    """
    A recursive depthwise-separable block with channel attention.
    Depth 1: DW Conv -> BN -> SiLU -> PW Conv -> BN -> Attention -> Residual
    Depth > 1: Parallel recursive branches (Left, Right) -> Average -> Base Processing
    """
    def __init__(self, depth, channels):
        super(FractalAttentionBlock, self).__init__()
        self.depth = depth
        self.channels = channels
        
        # Base processing components
        # Depthwise Conv
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        
        # Pointwise Conv
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Channel Attention (SE-like but lighter: GAP -> FC -> Sigmoid)
        # Using a small hidden layer to keep params low
        hidden_dim = max(8, channels // 8)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid()
        )
        
        # Recursive branches
        if depth > 1:
            self.left = FractalAttentionBlock(depth - 1, channels)
            self.right = FractalAttentionBlock(depth - 1, channels)
            
    def forward(self, x):
        # Recursive step: Process input through parallel branches and combine
        if self.depth > 1:
            left_out = self.left(x)
            right_out = self.right(x)
            # Deterministic average to stabilize 1-epoch training
            x = (left_out + right_out) / 2.0
            
        # Base processing
        # 1. Depthwise Conv
        y = self.dw_conv(x)
        y = self.bn1(y)
        y = self.act(y)
        
        # 2. Pointwise Conv
        y = self.pw_conv(y)
        y = self.bn2(y)
        
        # 3. Attention
        attn_weights = self.attention(x) # Apply attention to input x or y? Usually to the feature map.
        # Let's apply attention to the output of the conv block to modulate it
        # Actually, standard SE applies to the feature map. 
        # y is the feature map.
        y = y * attn_weights.unsqueeze(-1).unsqueeze(-1)
        
        # 4. Residual
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 64
        # Slightly larger kernel for initial context
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 64 channels, Depth 2 Fractal
        self.stage1 = FractalAttentionBlock(depth=2, channels=64)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 64 -> 128
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 128 channels, Depth 3 Fractal
        self.stage2 = FractalAttentionBlock(depth=3, channels=128)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 128 -> 256
        self.trans2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 256 channels, Depth 3 Fractal
        self.stage3 = FractalAttentionBlock(depth=3, channels=256)
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, n_classes)
        )
        
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
        # Apply normalization if needed
        # CIFAR-10 mean/std
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW as in Model 1
        # Map momentum to beta1
        beta1 = prm['momentum']
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(beta1, 0.999),
            weight_decay=1e-4
        )
        
        self.batch_size = prm.get('batch', 128)
        return self.optimizer
    
    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
        
        # Cosine schedule within 1 epoch
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