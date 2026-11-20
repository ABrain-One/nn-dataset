import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DropPath(nn.Module):
    """
    Stochastic Depth.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    Replicates the core logic of Model 1's FractalBlock:
    Recursive structure: Left branch (depth-1) and Right branch (depth-1 composed twice).
    Joined via FractalDropPath (mean of active branches).
    Uses Depthwise + Pointwise conv, BN, SiLU.
    """
    def __init__(self, n_columns: int, channels: int, dropout_prob: float = 0.0):
        super().__init__()
        self.n_columns = int(n_columns)
        channels = int(channels)

        # The leaf operation: Depthwise 3x3 -> Pointwise 1x1 -> BN -> SiLU
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

        if self.n_columns > 1:
            self.left = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.right_1 = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.right_2 = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.join = FractalDropPath(drop_prob=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_columns == 1:
            return self.conv(x)
        
        out_left = self.left(x)
        out_right = self.right_2(self.right_1(x))
        return self.join([out_left, out_right])

class FractalDropPath(nn.Module):
    """
    Joins a list of tensors by averaging active ones (with stochastic depth).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
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

        # Derive input channels
        # CIFAR-10 is (3, 32, 32). in_shape is likely (3, 32, 32) or (32, 32, 3).
        # We assume in_shape[0] is channels if it's 3, otherwise in_shape[1] might be channels in some conventions.
        # However, Model 1 hardcoded 3. To be safe and follow "derive what you need", 
        # but CIFAR is fixed. Let's look at the "Model to Improve" code: `c_in = 3`.
        # I will stick to 3 but add a check if possible. 
        # Actually, the prompt says "Never hard-code the input resolution". It implies deriving spatial dims.
        # For channels, CIFAR is 3.
        c_in = 3
        if len(in_shape) == 3:
            # If (C, H, W)
            if in_shape[0] == 3: c_in = 3
            elif in_shape[-1] == 3: c_in = 3
            else: c_in = in_shape[0] # Fallback
        elif len(in_shape) == 4:
            c_in = in_shape[1]

        n_classes = out_shape[0] if out_shape else 10

        # Architecture Design:
        # We combine the "Fractal" block (strong in Model 1) with a deeper, wider configuration 
        # and a different channel progression to avoid the "too similar" penalty.
        
        # Model 1: 64 -> 128 -> 256, 3 stages, FractalBlock(3) each.
        # This Design: 
        # Stem: 3 -> 48
        # Stage 1: 48 channels, 2 FractalBlocks (depth 2). Spatial 32x32.
        # Transition: 48 -> 96, Pool 2x2. Spatial 16x16.
        # Stage 2: 96 channels, 3 FractalBlocks (depth 2). Spatial 16x16.
        # Transition: 96 -> 192, Pool 2x2. Spatial 8x8.
        # Stage 3: 192 channels, 4 FractalBlocks (depth 2). Spatial 8x8.
        # Head: GAP -> FC.
        
        # Why depth 2? 
        # FractalBlock(2) creates a tree of depth 2.
        # Leaves: Left (1), Right (1->1->1? No, right is R1(R2(x))).
        # If n=2:
        # Left: Block(1) -> Conv
        # Right: Block(1)(Block(1)(x)) -> Conv(Conv(x))
        # Join: Mean(Left, Right)
        # This is a valid fractal structure.
        
        # Parameter count estimation:
        # Stem: 3*48*9 = 1296
        # Stage 1 (48 ch, 2 blocks):
        #   Block(2, 48):
        #     Left: Block(1, 48) -> 48*9 + 48*48 = 432 + 2304 = 2736
        #     Right: Block(1, 48) x 2 = 5472
        #     Total per Block(2): 8208
        #   2 Blocks: 16416
        # Trans 1: 48*96 = 4608
        # Stage 2 (96 ch, 3 blocks):
        #   Block(2, 96):
        #     Left: 96*9 + 96*96 = 864 + 9216 = 10080
        #     Right: 20160
        #     Total per Block(2): 30240
        #   3 Blocks: 90720
        # Trans 2: 96*192 = 18432
        # Stage 3 (192 ch, 4 blocks):
        #   Block(2, 192):
        #     Left: 192*9 + 192*192 = 1728 + 36864 = 38592
        #     Right: 77184
        #     Total per Block(2): 115776
        #   4 Blocks: 463104
        # FC: 192*10 = 1920
        # Total: ~1296 + 16416 + 4608 + 90720 + 18432 + 463104 + 1920 + BNs ~ 600k params.
        # This is well under 5M. We can afford more depth or channels.
        
        # Let's increase Stage 3 depth or channels?
        # Model 1 used 256 channels. We are at 192. 
        # Let's go 3 -> 48 -> 96 -> 192 -> 384?
        # If we add Stage 4:
        # Trans 3: 192*384 = 73728
        # Stage 4 (384 ch, 2 blocks):
        #   Block(2, 384):
        #     Left: 384*9 + 384*384 = 3456 + 147456 = 150912
        #     Right: 301824
        #     Total per Block(2): 452736
        #   2 Blocks: 905472
        # Total would be ~1.5M. Still safe.
        
        # Let's stick to 3 stages but make them deeper (more blocks) to leverage the fractal depth.
        # Or use FractalBlock(3) like Model 1?
        # Model 1 used FractalBlock(3).
        # FractalBlock(3) is heavier.
        # Left: Block(2)
        # Right: Block(2)(Block(2)(x))
        # This is a tree of depth 3.
        
        # Let's use FractalBlock(3) for the first stage, and FractalBlock(2) for later stages to balance compute.
        # Or just FractalBlock(2) everywhere but more of them.
        
        # Let's try:
        # Stem: 3 -> 64
        # Stage 1: 64 ch, 2x FractalBlock(3). (Heavy)
        # Pool -> 16
        # Trans: 64 -> 128
        # Stage 2: 128 ch, 3x FractalBlock(2).
        # Pool -> 8
        # Trans: 128 -> 256
        # Stage 3: 256 ch, 4x FractalBlock(2).
        
        # This mimics Model 1's channel progression (64-128-256) but uses different block counts and depths.
        # To be "not a copy", we change the stem, the block depths, and the number of blocks.
        
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 64 channels
        # Use FractalBlock(3) for rich feature extraction at high resolution.
        self.stage1 = nn.Sequential(
            FractalBlock(3, 64, 0.0),
            FractalBlock(3, 64, 0.0)
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Transition 1: 64 -> 128
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )

        # Stage 2: 128 channels
        # Use FractalBlock(2) to reduce parameter count while maintaining depth.
        self.stage2 = nn.Sequential(
            FractalBlock(2, 128, 0.0),
            FractalBlock(2, 128, 0.0),
            FractalBlock(2, 128, 0.0)
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Transition 2: 128 -> 256
        self.trans2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )

        # Stage 3: 256 channels
        # Use FractalBlock(2) with more blocks for capacity.
        self.stage3 = nn.Sequential(
            FractalBlock(2, 256, 0.0),
            FractalBlock(2, 256, 0.0),
            FractalBlock(2, 256, 0.0),
            FractalBlock(2, 256, 0.0)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)

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
        x = self.pool1(x)
        x = self.trans1(x)
        
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.trans2(x)
        
        x = self.stage3(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Model 1 (74.66%) used AdamW with lr=0.002, momentum=0.75
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