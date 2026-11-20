import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
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

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class MBConvBlock(nn.Module):
    """
    MobileNetV2 style block with a gating mechanism on the output projection.
    Structure: 
    1x1 expand -> SiLU -> 3x3 depthwise -> SiLU -> 1x1 project -> BN -> Sigmoid Gate -> Add
    """
    def __init__(self, in_ch, out_ch, expand_ratio=4, drop_path_prob=0.0):
        super().__init__()
        self.skip_connect = (in_ch == out_ch)
        
        hidden_dim = int(in_ch * expand_ratio)
        
        # Expansion
        self.expand = ConvBNAct(in_ch, hidden_dim, kernel_size=1, stride=1, padding=0)
        
        # Depthwise
        self.dw = ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim)
        
        # Projection
        self.project = nn.Conv2d(hidden_dim, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_proj = nn.BatchNorm2d(out_ch)
        
        # Gating: Learnable scalar per channel (or spatial? Let's stick to channel-wise 1x1 conv for gating)
        # To make it distinct and effective, we use a 1x1 conv to generate the gate.
        self.gate = nn.Conv2d(hidden_dim, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        identity = x
        
        y = self.expand(x)
        y = self.dw(y)
        
        # Project
        proj = self.project(y)
        proj = self.bn_proj(proj)
        
        # Gate
        g = torch.sigmoid(self.gate(y))
        
        # Apply gate to projection
        gated_proj = proj * g
        
        # Residual
        if self.skip_connect:
            out = x + self.drop_path(gated_proj)
        else:
            out = self.drop_path(gated_proj)
            
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        c_in = in_shape[1]
        n_classes = out_shape[0]
        
        # Architecture Design:
        # We combine the "Fractal" parallelism idea (from Model 1) with efficient depthwise convs (MobileNet style) 
        # and gating (from Design 2, but improved).
        # 
        # Key differentiator: 
        # - We use a "Dual-Path" structure in the final stage where two parallel MBConv blocks merge, 
        #   mimicking the FractalBlock join but with learned gating instead of simple averaging.
        # - We keep spatial resolution higher for longer (only 2 downsamplings).
        # - We use a unique channel progression: 48 -> 96 -> 192.
        
        # Stem: 3 -> 48
        self.stem = ConvBNAct(c_in, 48, kernel_size=3, stride=1, padding=1)
        
        # Stage 1: 48 channels, 32x32
        # 2 MBConv blocks
        self.stage1 = nn.Sequential(
            MBConvBlock(48, 48, expand_ratio=4, drop_path_prob=0.0),
            MBConvBlock(48, 48, expand_ratio=4, drop_path_prob=0.0),
        )
        
        # Transition 1: 48 -> 96, 32x32 -> 16x16
        self.trans1 = ConvBNAct(48, 96, kernel_size=1, stride=2, padding=0)
        
        # Stage 2: 96 channels, 16x16
        # 3 MBConv blocks
        self.stage2 = nn.Sequential(
            MBConvBlock(96, 96, expand_ratio=4, drop_path_prob=0.0),
            MBConvBlock(96, 96, expand_ratio=4, drop_path_prob=0.0),
            MBConvBlock(96, 96, expand_ratio=4, drop_path_prob=0.0),
        )
        
        # Transition 2: 96 -> 192, 16x16 -> 8x8
        self.trans2 = ConvBNAct(96, 192, kernel_size=1, stride=2, padding=0)
        
        # Stage 3: 192 channels, 8x8
        # This is where we introduce the "Fractal-like" dual path.
        # We split the input into two parallel branches, process them, and merge.
        # Branch A: 2 MBConv blocks
        # Branch B: 2 MBConv blocks
        # Merge: 1x1 Conv + BN + SiLU
        
        self.branch_a = nn.Sequential(
            MBConvBlock(192, 192, expand_ratio=4, drop_path_prob=0.0),
            MBConvBlock(192, 192, expand_ratio=4, drop_path_prob=0.0),
        )
        
        self.branch_b = nn.Sequential(
            MBConvBlock(192, 192, expand_ratio=4, drop_path_prob=0.0),
            MBConvBlock(192, 192, expand_ratio=4, drop_path_prob=0.0),
        )
        
        # Merge block: Takes sum of branches (or concat? Sum is cheaper and works well with BN)
        # Let's use a simple addition followed by a projection to stabilize.
        self.merge_proj = ConvBNAct(192, 192, kernel_size=1, stride=1, padding=0)
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, n_classes)
        
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
        
        # Dual Path Stage
        path_a = self.branch_a(x)
        path_b = self.branch_b(x)
        
        # Merge
        merged = path_a + path_b
        out = self.merge_proj(merged)
        
        x = self.gap(out)
        x = x.flatten(1)
        x = self.fc(x)
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