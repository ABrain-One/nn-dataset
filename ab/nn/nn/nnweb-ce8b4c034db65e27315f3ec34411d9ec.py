import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1. - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        random_tensor.div_(keep_prob)
        return x * random_tensor

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_bn=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block:
    1. Pointwise conv (1x1) to expand channels (if expand_ratio > 1)
    2. Large depthwise conv (7x7) for spatial mixing
    3. Pointwise conv (1x1) to project back
    4. LayerNorm (applied on flattened spatial dims, then restored)
    5. GELU activation
    6. Residual connection with DropPath
    """
    def __init__(self, dim, drop_path_prob=0.0, expand_ratio=4):
        super().__init__()
        self.dim = dim
        hidden_dim = int(dim * expand_ratio)
        
        # 1x1 expand
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim, bias=False)
        
        # 1x1 project back
        self.pw_conv = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        
        # LayerNorm on channels (applied after flattening spatial dims in forward)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        
        self.drop_path = DropPath(drop_path_prob)
        
        # Note: We need a 1x1 conv to expand before the DW conv if expand_ratio > 1
        # Standard ConvNeXt does: DW -> LN -> Act -> PW
        # But to match "Inverted Bottleneck" style efficiency while using 7x7:
        # We can do: 1x1 Expand -> DW 7x7 -> LN -> Act -> 1x1 Project -> Residual
        
        # Let's stick to the canonical ConvNeXt V1 block structure for stability:
        # X -> DW Conv (7x7) -> LayerNorm -> GELU -> Pointwise Conv (1x1) -> Residual
        # However, standard ConvNeXt assumes channels are already at the target width.
        # To save params and follow the "Efficient" theme of Design 1, I will use an inverted structure
        # but with the 7x7 kernel from ConvNeXt.
        
        # Revised Block:
        # X -> 1x1 Expand (to hidden_dim) -> DW 7x7 (hidden_dim) -> LN -> GELU -> 1x1 Project (to dim) -> Residual
        
        self.expand_conv = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False) if expand_ratio > 1 else nn.Identity()
        
    def forward(self, x):
        shortcut = x
        
        # Expand
        y = self.expand_conv(x)
        
        # Large Depthwise Conv
        y = self.dw_conv(y)
        
        # LayerNorm (requires flattening spatial dims)
        B, C, H, W = y.shape
        y = y.permute(0, 2, 3, 1) # B, H, W, C
        y = self.norm(y)
        y = self.act(y)
        y = y.permute(0, 3, 1, 2) # B, C, H, W
        
        # Project
        y = self.pw_conv(y)
        
        # Residual
        y = self.drop_path(y)
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        
        # Determine input channels
        # CIFAR-10 is typically (3, 32, 32) or (32, 32, 3).
        # The prompt says "Never hard-code the input resolution".
        # We assume standard CIFAR channel count is 3.
        c_in = 3
        if len(in_shape) == 3:
            if in_shape[0] == 3: c_in = 3
            elif in_shape[-1] == 3: c_in = 3
            else: c_in = in_shape[1]
        elif len(in_shape) == 4:
            c_in = in_shape[1]
            
        n_classes = out_shape[0] if out_shape else 10
        
        # Architecture:
        # Stem: 3 -> 48 (3x3, stride 1)
        # Stage 1: 48 channels, 32x32 -> 2 ConvNeXt-like blocks (7x7 DW)
        # Downsample: 48 -> 96, 32 -> 16 (AvgPool 2)
        # Stage 2: 96 channels, 16x16 -> 2 ConvNeXt-like blocks
        # Downsample: 96 -> 192, 16 -> 8 (AvgPool 2)
        # Stage 3: 192 channels, 8x8 -> 2 ConvNeXt-like blocks
        # Head: GAP -> FC
        
        # This uses the 7x7 depthwise kernel from ConvNeXt (Model 4 influence, though Model 4 was DarkNet, 
        # but 7x7 DW is a strong modern block) combined with the channel progression from Design 1 (48, 96, 192) 
        # to avoid the 64-128-256-512 pattern.
        
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(48, drop_path_prob=0.0, expand_ratio=4),
            ConvNeXtBlock(48, drop_path_prob=0.0, expand_ratio=4)
        )
        
        # Transition 1: 48 -> 96, 32 -> 16
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 channels
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(96, drop_path_prob=0.0, expand_ratio=4),
            ConvNeXtBlock(96, drop_path_prob=0.0, expand_ratio=4)
        )
        
        # Transition 2: 96 -> 192, 16 -> 8
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 192 channels
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(192, drop_path_prob=0.0, expand_ratio=4),
            ConvNeXtBlock(192, drop_path_prob=0.0, expand_ratio=4)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(192, n_classes)
        
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        
        x = self.stage1(x)
        x = self.trans1(x)
        
        x = self.stage2(x)
        x = self.trans2(x)
        
        x = self.stage3(x)
        
        x = self.global_pool(x)
        x = x.flatten(1)
        
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
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
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            
            self.optimizer.step()
            scheduler.step()