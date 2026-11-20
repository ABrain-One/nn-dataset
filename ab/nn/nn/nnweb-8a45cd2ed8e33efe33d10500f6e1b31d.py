import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class FractalBlock(nn.Module):
    """
    A recursive block that creates a fractal-like structure of depthwise and pointwise convolutions.
    It uses a 'join' operation that averages the outputs of parallel branches (with optional stochastic depth).
    """
    def __init__(self, n_columns: int, channels: int, dropout_prob: float = 0.0):
        super().__init__()
        self.n_columns = int(n_columns)
        channels = int(channels)

        # Depthwise + Pointwise Convolution (Efficient Block core)
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

    def forward(self, x):
        if self.n_columns == 1:
            return self.conv(x)
        out_left = self.left(x)
        out_right = self.right_2(self.right_1(x))
        return self.join([out_left, out_right])

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

        # Derive input channels
        # Based on Model 1 and 2, in_shape[1] is likely channels if format is (N, C, H, W)
        # But Model 1 hardcodes 3. To be safe and follow "Never hard-code input resolution",
        # we should derive channels if possible, but CIFAR is always 3.
        # However, the prompt says "Never hard-code the input resolution". It doesn't explicitly forbid hardcoding channels,
        # but robust code should derive it.
        # Let's look at the "Model to Improve" code: `c_in = 3`.
        # I will follow the robust pattern: if len(in_shape)==3, assume (C, H, W) or (H, W, C)?
        # Model 2 uses `in_shape[1]`. If in_shape is (3, 32, 32), this is 32. That would be wrong.
        # If in_shape is (32, 32, 3), this is 32.
        # If in_shape is (N, C, H, W), this is C.
        # Given the ambiguity and that CIFAR is fixed at 3 channels, I will use `in_shape[1]` but add a check.
        # Actually, looking at the provided strong models, they seem to assume `in_shape` might be (N, C, H, W) or similar.
        # Let's assume `in_shape` is (C, H, W) or (H, W, C).
        # If I use `in_shape[0]` for channels, and it's (H, W, C), it's wrong.
        # If I use `in_shape[-1]`, and it's (C, H, W), it's wrong.
        # The safest bet for CIFAR-10 in these challenges is often that `in_shape` is (3, 32, 32) or (32, 32, 3).
        # Let's look at Model 1 again. It ignores in_shape for channels.
        # I will use `in_shape[1]` as channels, assuming the convention is (N, C, H, W) or that the harness passes (C, H, W) where C is at index 1? No, that's unlikely.
        # Let's try to be smart: if any dimension is 3, it's likely channels.
        c_in = 3
        if len(in_shape) == 3:
            if in_shape[0] == 3: c_in = 3
            elif in_shape[-1] == 3: c_in = 3
            else: c_in = in_shape[1] # Fallback
        elif len(in_shape) == 4:
            c_in = in_shape[1]

        n_classes = out_shape[0] if out_shape else 10

        # Architecture Design:
        # 1. Stem: Conv 3x3 -> BN -> SiLU (Channels: 3 -> 48)
        # 2. Stage 1: FractalBlock (depth 2) at 48 channels. Spatial: 32x32
        # 3. Pool: AvgPool 2x2 -> 16x16
        # 4. Stage 2: FractalBlock (depth 2) at 96 channels. (Transition 48->96)
        # 5. Pool: AvgPool 2x2 -> 8x8
        # 6. Stage 3: FractalBlock (depth 2) at 192 channels. (Transition 96->192)
        # 7. Head: Global Avg Pool -> Linear
        
        # Parameter Budget:
        # Stem: 3*48*3*3 = 1296
        # Stage 1 (48 ch, depth 2):
        #   FractalBlock(2, 48):
        #     Left: FractalBlock(1, 48) -> Conv(48,48,3x3,dw) + Conv(48,48,1x1) = 48*9 + 48*48 = 432 + 2304 = 2736
        #     Right: FractalBlock(1, 48) x 2 = 2 * 2736 = 5472
        #     Total Stage 1 Block: 2736 + 5472 = 8208
        #   We use 1 such block.
        # Transition 1: Conv(48, 96, 1x1) = 48*96 = 4608
        # Stage 2 (96 ch, depth 2):
        #   FractalBlock(2, 96):
        #     Left: FractalBlock(1, 96) -> 96*9 + 96*96 = 864 + 9216 = 10080
        #     Right: 2 * 10080 = 20160
        #     Total: 30240
        #   We use 2 such blocks? Let's use 1 to save params.
        # Transition 2: Conv(96, 192, 1x1) = 96*192 = 18432
        # Stage 3 (192 ch, depth 2):
        #   FractalBlock(2, 192):
        #     Left: FractalBlock(1, 192) -> 192*9 + 192*192 = 1728 + 36864 = 38592
        #     Right: 2 * 38592 = 77184
        #     Total: 115776
        #   We use 1 such block.
        # Head: Linear(192, 10) = 1920
        # Total approx: 1.3k + 8.2k + 4.6k + 30.2k + 18.4k + 115.8k + 1.9k ~ 180k params. Very small.
        # We can afford deeper fractals or more blocks.
        
        # Let's increase depth and channels to improve accuracy.
        # Stem: 3 -> 64
        # Stage 1: 2 FractalBlocks (depth 2) at 64 ch.
        # Pool -> 16x16
        # Transition: 64 -> 128
        # Stage 2: 2 FractalBlocks (depth 2) at 128 ch.
        # Pool -> 8x8
        # Transition: 128 -> 256
        # Stage 3: 2 FractalBlocks (depth 2) at 256 ch.
        # Head: GAP -> Linear(256, 10)

        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 64 channels
        self.stage1 = nn.Sequential(
            FractalBlock(2, 64, 0.0),
            FractalBlock(2, 64, 0.0)
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Transition 1: 64 -> 128
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )

        # Stage 2: 128 channels
        self.stage2 = nn.Sequential(
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
        self.stage3 = nn.Sequential(
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
        # We use AdamW as per the fixed training setup of the model to improve.
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