import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        random_tensor.div_(keep_prob)
        return x * random_tensor

class GatedSeparableBlock(nn.Module):
    """
    A hybrid block combining:
    1. Inverted Bottleneck structure (Expansion -> Depthwise -> Projection) from EfficientNet-like designs (Design 1).
    2. Gated Residual mechanism (Sigmoid gate on the residual stream) from Design 2.
    3. SiLU activation and BatchNorm.
    
    This aims to capture the expressiveness of the fractal/efficient blocks but with a cleaner, non-recursive, non-standard ResNet layout.
    """
    def __init__(self, channels, expand_ratio=4, drop_prob=0.0):
        super().__init__()
        self.channels = channels
        hidden_dim = int(channels * expand_ratio)
        
        # Expansion: 1x1 conv
        self.expand_conv = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise: 3x3 depthwise conv
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_dim)
        
        # Projection: 1x1 conv
        self.proj_conv = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(channels)
        
        self.act = nn.SiLU(inplace=True)
        self.drop_path = DropPath(drop_prob)
        
        # Gating mechanism: 1x1 conv to generate gate
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gate_act = nn.Sigmoid()

    def forward(self, x):
        identity = x
        
        # Main path
        y = self.expand_conv(x)
        y = self.expand_bn(y)
        y = self.act(y)
        
        y = self.dw_conv(y)
        y = self.dw_bn(y)
        y = self.act(y)
        
        y = self.proj_conv(y)
        y = self.proj_bn(y)
        
        # Apply gating
        gate = self.gate_conv(x)
        gate = self.gate_act(gate)
        
        # Combine: Identity + (Processed * Gate)
        # Note: Standard residual adds y. Here we add y * gate.
        # This allows the network to learn to "turn off" certain features in the residual stream.
        y = y * gate
        
        y = self.drop_path(y)
        
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        
        c_in = 3
        if len(in_shape) == 3:
            if in_shape[0] == 3: c_in = 3
            elif in_shape[-1] == 3: c_in = 3
            else: c_in = in_shape[1]
        elif len(in_shape) == 4:
            c_in = in_shape[1]
            
        n_classes = out_shape[0] if out_shape else 10
        
        # Architecture Configuration
        # Goal: Beat 77.14% (Design 1) and 74.34% (Model to Improve).
        # Design 1 used: 32->64->128 channels, 2/3/4 blocks.
        # Model to Improve used: 64->128->256 channels, 3 fractal blocks.
        
        # New Design:
        # Stem: 3 -> 48
        # Stage 1: 48 channels, 3 blocks. Spatial 32x32.
        # Transition: 48 -> 96 (1x1 conv + AvgPool 2x2). Spatial 16x16.
        # Stage 2: 96 channels, 4 blocks.
        # Transition: 96 -> 192 (1x1 conv + AvgPool 2x2). Spatial 8x8.
        # Stage 3: 192 channels, 4 blocks.
        # Head: GAP -> FC.
        
        # Parameter Estimate:
        # Stem: 3*48*9 = 1296
        # Stage 1 (3 blocks @ 48ch):
        #   Block: Expand(48->192, 1x1) + DW(192, 3x3) + Proj(192->48, 1x1) + Gate(48->48, 1x1)
        #   Expand: 48*192 = 9216
        #   DW: 192*9 = 1728
        #   Proj: 192*48 = 9216
        #   Gate: 48*48 = 2304
        #   BNs: ~4 * 96 = 384
        #   Per Block: ~22848
        #   3 Blocks: ~68544
        # Transition 1: 48*96 = 4608
        # Stage 2 (4 blocks @ 96ch):
        #   Expand: 96*384 = 36864
        #   DW: 384*9 = 3456
        #   Proj: 384*96 = 36864
        #   Gate: 96*96 = 9216
        #   BNs: ~4 * 192 = 768
        #   Per Block: ~87168
        #   4 Blocks: ~348672
        # Transition 2: 96*192 = 18432
        # Stage 3 (4 blocks @ 192ch):
        #   Expand: 192*768 = 147456
        #   DW: 768*9 = 6912
        #   Proj: 768*192 = 147456
        #   Gate: 192*192 = 36864
        #   BNs: ~4 * 384 = 1536
        #   Per Block: ~338224
        #   4 Blocks: ~1352896
        # Head: 192*10 = 1920
        # Total: ~1.8M params. Well within 5M limit.
        
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels
        self.stage1 = nn.Sequential(
            GatedSeparableBlock(48, expand_ratio=4, drop_prob=0.0),
            GatedSeparableBlock(48, expand_ratio=4, drop_prob=0.0),
            GatedSeparableBlock(48, expand_ratio=4, drop_prob=0.0),
        )
        
        # Transition 1: 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 channels
        self.stage2 = nn.Sequential(
            GatedSeparableBlock(96, expand_ratio=4, drop_prob=0.0),
            GatedSeparableBlock(96, expand_ratio=4, drop_prob=0.0),
            GatedSeparableBlock(96, expand_ratio=4, drop_prob=0.0),
            GatedSeparableBlock(96, expand_ratio=4, drop_prob=0.0),
        )
        
        # Transition 2: 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 192 channels
        self.stage3 = nn.Sequential(
            GatedSeparableBlock(192, expand_ratio=4, drop_prob=0.0),
            GatedSeparableBlock(192, expand_ratio=4, drop_prob=0.0),
            GatedSeparableBlock(192, expand_ratio=4, drop_prob=0.0),
            GatedSeparableBlock(192, expand_ratio=4, drop_prob=0.0),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(192, n_classes)
        
        self._init_weights()
        self.to(device)

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
        # Using the same optimizer setup as the model to improve (which is fixed)
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