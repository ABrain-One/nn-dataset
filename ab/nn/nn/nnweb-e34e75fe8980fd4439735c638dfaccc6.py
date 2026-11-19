import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class FractalBlock(nn.Module):
    """
    A recursive block that creates a fractal-like structure of depthwise and pointwise convolutions.
    It uses a 'join' operation that averages the outputs of parallel branches (with optional stochastic depth).
    This is adapted from Model 1 but modified to include residual connections at the base level
    to stabilize training and improve feature reuse.
    """
    def __init__(self, n_columns: int, channels: int, dropout_prob: float = 0.0):
        super().__init__()
        self.n_columns = int(n_columns)
        channels = int(channels)

        # Depthwise + Pointwise Convolution (Efficient Block core)
        # We add a residual connection here to the base conv to ensure gradient flow
        # and to make the block identity-friendly.
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # Activation after the second conv? Or before? 
        # Model 1 did: DW -> PW -> BN -> SiLU. 
        # Let's stick to DW -> BN -> SiLU -> PW -> BN. 
        # Actually, standard EfficientNet is DW -> BN -> Act -> PW -> BN.
        # Let's refine the base conv to be: DW -> BN -> SiLU -> PW -> BN.
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        if self.n_columns > 1:
            self.left = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.right_1 = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.right_2 = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.join = FractalDropPath(drop_prob=dropout_prob)
            
            # We need a way to combine the outputs. 
            # In Model 1, it was just a mean. 
            # Let's add a 1x1 conv to mix the joined features before adding to identity?
            # No, let's keep it simple: The output of the fractal structure is added to the input.
            # But the fractal structure itself doesn't have a direct identity path in Model 1.
            # Let's add an identity path for the whole block.
            
    def forward(self, x):
        if self.n_columns == 1:
            # Base case: Conv + Identity
            return x + self.conv(x)
        
        out_left = self.left(x)
        out_right = self.right_2(self.right_1(x))
        
        # Join the two branches
        joined = self.join([out_left, out_right])
        
        # Add the original input to the joined result to maintain residual flow
        # This makes the block stable.
        return x + joined

class FractalDropPath(nn.Module):
    """
    Stochastic Depth for a list of tensors.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        if not self.training:
            return torch.stack(inputs).mean(dim=0)
        n = len(inputs)
        mask = torch.bernoulli(torch.full((n,), 1 - self.drop_prob, device=inputs[0].device))
        if mask.sum() == 0:
            mask[torch.randint(0, n, (1,)).item()] = 1.0
        active = [inp for inp, m in zip(inputs, mask) if m > 0]
        return torch.stack(active).mean(dim=0)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device

        # Derive input channels robustly
        c_in = 3
        if len(in_shape) == 3:
            # Assume (C, H, W) or (H, W, C). CIFAR is 3.
            # If in_shape[0] is 3, it's likely C. If in_shape[-1] is 3, it's likely C.
            # Standard PyTorch is (N, C, H, W), but in_shape is often just (C, H, W) or (H, W, C) in these prompts.
            # Let's check for 3.
            if in_shape[0] == 3:
                c_in = 3
            elif in_shape[-1] == 3:
                c_in = 3
            else:
                c_in = in_shape[1] # Fallback
        elif len(in_shape) == 4:
            c_in = in_shape[1]

        n_classes = out_shape[0] if out_shape else 10

        # Architecture Design:
        # We combine the Fractal Block (from Model 1) with a more standard residual backbone.
        # We avoid the 64-128-256-512 progression.
        # We use a progression: 48 -> 96 -> 192 -> 384.
        # This is similar to Design 2 but using Fractal Blocks instead of Gated Residual Blocks.
        # The Fractal Blocks provide a unique, recursive feature extraction that is efficient.
        
        # Stem: 3 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, 32x32 -> 16x16
        # We use 2 FractalBlocks of depth 2.
        # Depth 2 means: Left(1) and Right(1, 1).
        # This creates a small tree.
        self.stage1_blocks = nn.Sequential(
            FractalBlock(2, 48, 0.0),
            FractalBlock(2, 48, 0.0)
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition 1: 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 96 channels, 16x16 -> 8x8
        self.stage2_blocks = nn.Sequential(
            FractalBlock(2, 96, 0.0),
            FractalBlock(2, 96, 0.0)
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition 2: 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 192 channels, 8x8 -> 4x4
        self.stage3_blocks = nn.Sequential(
            FractalBlock(2, 192, 0.0),
            FractalBlock(2, 192, 0.0)
        )
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition 3: 192 -> 384
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        # Stage 4: 384 channels, 4x4
        # We use 1 FractalBlock of depth 2 here to save params, or maybe depth 3?
        # Let's stick to depth 2 for consistency, maybe 2 blocks.
        self.stage4_blocks = nn.Sequential(
            FractalBlock(2, 384, 0.0),
            FractalBlock(2, 384, 0.0)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        # We add a small MLP head to help with feature separation
        self.classifier = nn.Sequential(
            nn.Linear(384, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes)
        )
        
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
        
        x = self.stage1_blocks(x)
        x = self.pool1(x)
        x = self.trans1(x)
        
        x = self.stage2_blocks(x)
        x = self.pool2(x)
        x = self.trans2(x)
        
        x = self.stage3_blocks(x)
        x = self.pool3(x)
        x = self.trans3(x)
        
        x = self.stage4_blocks(x)
        
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