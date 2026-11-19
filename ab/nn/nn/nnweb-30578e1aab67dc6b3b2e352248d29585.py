import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

# --- Helper Classes ---

class FractalDropPath(nn.Module):
    """
    Taken from Model 1 (Strongest).
    Averages a list of tensors, randomly dropping some during training.
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

class FractalBlock(nn.Module):
    """
    Taken from Model 1 (Strongest).
    Recursive structure: Left branch (depth-1) + Right branch (depth-2) -> Join.
    Base case: Depthwise 3x3 -> Pointwise 1x1 -> BN -> SiLU.
    """
    def __init__(self, n_columns: int, channels: int, dropout_prob: float):
        super().__init__()
        self.n_columns = int(n_columns)
        channels = int(channels)  

        activation_layer = nn.SiLU(inplace=True)
        # Depthwise separable conv: 3x3 depthwise + 1x1 pointwise
        conv_layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
        norm_layer = nn.BatchNorm2d(channels)

        self.conv = nn.Sequential(
            conv_layer,
            norm_layer,
            activation_layer
        )

        if self.n_columns > 1:
            # Recursively build sub-branches
            self.left = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.right_1 = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.right_2 = FractalBlock(self.n_columns - 1, channels, dropout_prob)
            self.join = FractalDropPath(drop_prob=dropout_prob)

    def forward(self, x):
        if self.n_columns == 1: 
            return self.conv(x)
        out_left = self.left(x)
        # Right branch is sequential composition of two sub-branches
        out_right = self.right_2(self.right_1(x))
        return self.join([out_left, out_right])

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device

        # Derive input channels safely
        # Based on Model 1 and 2, in_shape[1] is channels if (N,C,H,W) or (C,H,W) where C is 1st? 
        # Model 1 hardcodes 3. Model 2 uses in_shape[1]. 
        # To be robust: if in_shape has 3 elements, assume (C,H,W) or (H,W,C)? 
        # CIFAR is 3 channels. If in_shape[0]==3, it's (C,H,W). If in_shape[2]==3, it's (H,W,C).
        # If in_shape has 4 elements, assume (N,C,H,W).
        if len(in_shape) == 4:
            c_in = in_shape[1]
        elif len(in_shape) == 3:
            # Heuristic: if first is 3, it's channels. If last is 3, it's channels.
            if in_shape[0] == 3:
                c_in = 3
            elif in_shape[2] == 3:
                c_in = 3
            else:
                c_in = in_shape[0] # Fallback
        else:
            c_in = 3
            
        n_classes = out_shape[0] if out_shape else 10

        # Architecture Design:
        # Model 1 (74.66%) used: Entry(64) -> [FractalBlock(3) -> Pool -> Trans] x3.
        # It used ReLU in Entry/Trans and SiLU in Blocks.
        # It used AvgPool.
        
        # My Design:
        # 1. Stem: Conv 3x3 (stride 1) -> BN -> SiLU. (Start with 48 channels to be slightly lighter/different)
        # 2. Stage 1: 2 FractalBlocks (depth 3) at 48 channels.
        # 3. Transition: 1x1 Conv (stride 2) -> BN -> SiLU. (Downsample + Channel change)
        # 4. Stage 2: 2 FractalBlocks (depth 3) at 96 channels.
        # 5. Transition: 1x1 Conv (stride 2) -> BN -> SiLU.
        # 6. Stage 3: 2 FractalBlocks (depth 3) at 192 channels.
        # 7. Head: GAP -> FC.
        
        # Why this is better than Model 1:
        # - Model 1 used 64->128->256. I use 48->96->192.
        # - Model 1 used ReLU in stem/trans. I use SiLU everywhere (smoother).
        # - Model 1 had 3 stages of 1 block each. I have 3 stages of 2 blocks each (deeper, same width).
        # - Model 1 used AvgPool 2x2. I use Strided Conv 1x1 (learns feature selection, preserves more info).
        # - Parameter count:
        #   Stem: 3*48*3*3 + 48 = 1296
        #   Stage 1 (48 ch, 2 blocks, depth 3):
        #     Each FractalBlock(depth 3) has:
        #       Left (depth 2): Left(depth 1) + Right(depth 1+1)
        #       Depth 1: 2 convs (dw+pw) = 48*48*9 + 48*48*1 = 20736 + 2304 = 23040 params.
        #       Depth 2: 2*Depth1 = 46080 params.
        #       Depth 3: Left(D2) + Right(D2) = 92160 params.
        #     So 1 FractalBlock(depth 3) ~ 92k params.
        #     2 blocks = 184k.
        #   Stage 2 (96 ch): 4x params of stage 1 ~ 736k.
        #   Stage 3 (192 ch): 16x params of stage 1 ~ 2.9M.
        #   Total ~ 3.8M. Well within 5M.
        
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels
        self.stage1 = nn.Sequential(
            FractalBlock(3, 48, 0.0),
            FractalBlock(3, 48, 0.0)
        )
        
        # Transition 1: 48 -> 96, spatial /2
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 96 channels
        self.stage2 = nn.Sequential(
            FractalBlock(3, 96, 0.0),
            FractalBlock(3, 96, 0.0)
        )
        
        # Transition 2: 96 -> 192, spatial /2
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 192 channels
        self.stage3 = nn.Sequential(
            FractalBlock(3, 192, 0.0),
            FractalBlock(3, 192, 0.0)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, n_classes)
        
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
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
        # We use SGD with momentum because it is often more stable for 1-epoch training
        # when combined with a proper LR schedule.
        # Let's stick to the strong signal from Model 1: AdamW.
        # But Model 1's success might be due to the specific architecture + AdamW.
        # Let's try SGD with a cosine schedule, which is the gold standard for 1-epoch CIFAR.
        # Actually, let's look at the hyperparameters provided.
        # If I use SGD, I should use a high LR.
        # If I use AdamW, I should use a low LR.
        
        # Let's use SGD with momentum as it's robust.
        # But wait, Model 1 used AdamW.
        # Let's try a hybrid: AdamW with a cosine schedule.
        
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