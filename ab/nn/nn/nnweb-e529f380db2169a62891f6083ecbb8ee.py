import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum', 'transform'}

class ChannelMix(nn.Module):
    """1x1 Conv + BN + GELU to mix channels and adjust dimension."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class EfficientBlock(nn.Module):
    """
    Depthwise Separable Conv Block with residual connection.
    Input: (B, C_in, H, W)
    Output: (B, C_out, H, W)
    """
    def __init__(self, in_ch, out_ch, expand=4, drop_path_prob=0.0):
        super().__init__()
        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        
        mid_ch = out_ch * expand
        # Expansion: 1x1 conv
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )
        
        # Depthwise Conv
        self.dw = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )
        
        # Projection: 1x1 conv
        self.proj = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        
        # Stochastic Depth
        self.drop_path = drop_path_prob

    def forward(self, x):
        identity = x
        if self.skip is not None:
            identity = self.skip(x)
        
        out = self.expand(x)
        out = self.dw(out)
        out = self.proj(out)
        
        if self.drop_path > 0 and self.training:
            # Stochastic depth
            mask = torch.bernoulli(torch.full((x.size(0), 1, 1, 1), 1.0 - self.drop_path, device=x.device))
            out = out * mask / (1.0 - self.drop_path)
            
        return out + identity

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        # Derive dimensions
        # in_shape is (C, H, W) or (H, W, C)? Usually (C, H, W) in these prompts based on Model 2 using in_shape[1] as channels.
        # Let's assume standard PyTorch input shape for the tensor is (N, C, H, W).
        # The constructor receives in_shape. Let's inspect Model 2: `in_channels = in_shape[1]`.
        # This implies in_shape is likely (N, C, H, W) or just (C, H, W) where index 1 is C?
        # If in_shape is (3, 32, 32), index 1 is 32 (H). That would be wrong for Model 2.
        # If in_shape is (N, C, H, W), index 1 is C.
        # Let's look at Model 1: `c_in = 3`. It hardcodes 3.
        # Model 3: `self.conv1 = nn.Conv2d(in_shape[1], 64, ...)`.
        # If in_shape is (3, 32, 32), in_shape[1] is 32. Conv2d(32, 64) would fail on CIFAR-10 (3 channels).
        # Therefore, in_shape MUST be (N, C, H, W) or (C, H, W) where C is NOT at index 1?
        # Wait, standard CIFAR-10 tensor is (N, 3, 32, 32).
        # If in_shape is (3, 32, 32), then in_shape[0] is 3.
        # If Model 2 uses in_shape[1] as channels, it expects in_shape to be (N, C, H, W) where N is index 0?
        # Or maybe the prompt passes (C, H, W) and the models are buggy but worked?
        # Let's assume in_shape is (N, C, H, W) based on standard PyTorch conventions often used in these "in_shape" arguments.
        # However, to be safe, I will try to infer C.
        # Common convention in these specific eval harnesses: in_shape is (C, H, W) or (H, W, C)?
        # Let's look at Model 1 again. It ignores in_shape for channels (hardcodes 3).
        # Model 4: `in_channels = in_shape[1]`.
        # If I pass (3, 32, 32), in_shape[1] is 32.
        # If I pass (32, 32, 3), in_shape[1] is 32.
        # If I pass (1, 3, 32, 32), in_shape[1] is 3.
        # Given Model 2 and 4 use index 1, and they are "strong models" (presumably they ran successfully on the dataset), it is highly likely that `in_shape` is provided as a 4-tuple `(N, C, H, W)` or the harness passes `(C, H, W)` and the models are actually failing/wrong but the prompt says they are "strong classifiers... with their accuracy".
        # Actually, if they are "strong", they must have run. If they ran with `in_shape[1]` as channels, then `in_shape[1]` MUST be 3.
        # This implies `in_shape` is likely `(N, C, H, W)` or `(C, H, W)` where C is at index 1? No, (C,H,W) has C at 0.
        # Maybe `in_shape` is `(H, W, C)`? Then index 2 is C.
        # Maybe `in_shape` is `(C, H, W)` and the models are just buggy but the "accuracy" reported is from a corrected version?
        # Let's assume the standard: `in_shape` is `(C, H, W)` or `(N, C, H, W)`.
        # To be robust, I will check the length and values.
        # If len==3: likely (C, H, W) or (H, W, C). CIFAR is 3 channels.
        # If len==4: likely (N, C, H, W).
        
        # Let's look at Model 2 again. `in_channels = in_shape[1]`.
        # If the dataset is CIFAR-10, channels=3.
        # If `in_shape` is (3, 32, 32), `in_shape[1]` is 32. `Conv2d(32, 256)` would crash on input (N, 3, 32, 32).
        # Therefore, `in_shape` MUST be such that `in_shape[1]` is 3.
        # This implies `in_shape` is likely `(N, C, H, W)` -> `(1, 3, 32, 32)`.
        # I will assume `in_shape` is `(N, C, H, W)`.
        
        if len(in_shape) == 4:
            self.in_channels = in_shape[1]
            self.h = in_shape[2]
            self.w = in_shape[3]
        elif len(in_shape) == 3:
            # Guessing (C, H, W) or (H, W, C).
            # If (C, H, W), C=in_shape[0].
            # If (H, W, C), C=in_shape[2].
            # CIFAR-10 is square. 
            # Let's assume (C, H, W) is more common in torchvision transforms output shapes if N is stripped.
            # But Model 2 uses index 1. This is confusing.
            # I will bet on (N, C, H, W) being the standard for this specific harness given Model 2/4 behavior.
            # If it's 3-tuple, I'll assume (C, H, W) and take index 0.
            # Wait, if Model 2 works, and it uses index 1, and CIFAR has 3 channels, then index 1 MUST be 3.
            # So if 3-tuple, it must be (X, 3, Y) or (3, X, Y) where index 1 is 3? No, (3, X, Y) has index 0 as 3.
            # So it must be (N, C, H, W) where N is ignored or present.
            # I will handle 4-tuple as (N, C, H, W).
            # If 3-tuple, I will assume (C, H, W) and use index 0, but I'll add a fallback.
            # Actually, let's look at the "strong" claim. If Model 2 is 69%, it worked.
            # I will assume in_shape is (N, C, H, W).
            
            # Fallback logic:
            # If len==3, assume (C, H, W).
            self.in_channels = in_shape[0]
            self.h = in_shape[1]
            self.w = in_shape[2]
        else:
            # Default to CIFAR-10
            self.in_channels = 3
            self.h = 32
            self.w = 32

        n_classes = out_shape[0] if out_shape else 10

        # Architecture:
        # Stem: 3x3 Conv, BN, GELU
        # Stage 1: 2x EfficientBlock (C=64), AvgPool
        # Stage 2: 2x EfficientBlock (C=128), AvgPool
        # Stage 3: 2x EfficientBlock (C=256), AvgPool
        # Stage 4: 1x EfficientBlock (C=512)
        # Global Pool + FC
        
        # This avoids 64-128-256-512 progression? 
        # Rule: "Do not use the 64-128-256-512 channel progression".
        # I will use: 48, 96, 192, 384.
        
        stem_ch = 48
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, stem_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.GELU()
        )
        
        self.stage1 = nn.Sequential(
            EfficientBlock(stem_ch, 48, expand=4, drop_path_prob=0.0),
            EfficientBlock(48, 48, expand=4, drop_path_prob=0.05),
            nn.AvgPool2d(2)
        )
        
        self.stage2 = nn.Sequential(
            EfficientBlock(48, 96, expand=4, drop_path_prob=0.1),
            EfficientBlock(96, 96, expand=4, drop_path_prob=0.15),
            nn.AvgPool2d(2)
        )
        
        self.stage3 = nn.Sequential(
            EfficientBlock(96, 192, expand=4, drop_path_prob=0.2),
            EfficientBlock(192, 192, expand=4, drop_path_prob=0.25),
            nn.AvgPool2d(2)
        )
        
        self.stage4 = nn.Sequential(
            EfficientBlock(192, 384, expand=4, drop_path_prob=0.3),
            EfficientBlock(384, 384, expand=4, drop_path_prob=0.35)
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(384, n_classes)
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # Use SGD with momentum. 
        # LR schedule: Cosine decay within the epoch.
        # We need to know the number of steps.
        # We'll estimate based on dataset size / batch size.
        # Since we don't have dataset size in train_setup, we'll assume standard CIFAR-10 (50,000).
        # But `train_data` is passed to `learn`.
        # We can set up the scheduler in `learn` or use a generic one.
        # Let's use a manual cosine decay in `learn` for better control over the single epoch.
        
        lr = prm['lr']
        momentum = prm['momentum']
        
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=1e-4
        )
        
        # We'll define the total steps in learn
        self.max_steps = None
        self.base_lr = lr
        self.momentum = momentum
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        # Estimate total steps
        # We can't easily get len(train_data) if it's a generator, but usually it's a DataLoader.
        # Let's try to get the length.
        try:
            total_steps = len(train_data)
        except:
            total_steps = 50000 // 128 # Fallback
        
        # Cosine schedule
        def cosine_lr(step):
            import math
            # Warmup for first 5%
            warmup_steps = max(1, int(0.05 * total_steps))
            if step < warmup_steps:
                return self.base_lr * (step / warmup_steps)
            
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        step = 0
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Update LR
            current_lr = cosine_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            step += 1
            
            # Optional: break if we hit max steps (safety)
            if step >= total_steps:
                break