import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, act=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class FractalBlock(nn.Module):
    """
    A block that processes input through two parallel branches:
    1. A depthwise-separable branch (Depthwise 3x3 -> Pointwise 1x1)
    2. A standard 3x3 branch
    The outputs are summed (or averaged) to create a rich feature representation.
    This mimics the 'Fractal' concept of recursive/parallel processing but with a simpler,
    more efficient structure than the recursive tree in Model 1.
    """
    def __init__(self, channels, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
        # Branch 1: Depthwise Separable (Efficient)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(channels)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(channels)
        
        # Branch 2: Standard 3x3 (Receptive Field)
        self.std_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.std_bn = nn.BatchNorm2d(channels)
        
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # Branch 1
        y1 = self.dw_conv(x)
        y1 = self.dw_bn(y1)
        y1 = self.pw_conv(y1)
        y1 = self.pw_bn(y1)
        
        # Branch 2
        y2 = self.std_conv(x)
        y2 = self.std_bn(y2)
        
        # Combine (Average to maintain scale stability, similar to FractalDropPath mean)
        y = (y1 + y2) / 2.0
        
        # Stochastic Depth (Simple implementation)
        if self.drop_prob > 0 and self.training:
            mask = torch.bernoulli(torch.full((y.shape[0], 1, 1, 1), 1 - self.drop_prob, device=y.device))
            y = y * mask / (1 - self.drop_prob)
            
        return self.act(y)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        
        # Derive input channels
        # Based on previous designs, in_shape is likely (C, H, W) or (N, C, H, W).
        # Model 1 hardcodes 3. Model 2 uses in_shape[1].
        # To be safe and consistent with the "Model to Improve" which hardcodes c_in=3,
        # but respecting the "derive what you need" rule, I'll try to be robust.
        # However, CIFAR-10 is always 3 channels. I will use in_shape[1] if it looks like channels, else 3.
        # Actually, looking at the provided code, `in_shape` is passed.
        # Let's assume standard (C, H, W) or (N, C, H, W).
        # If len(in_shape) == 3, assume (C, H, W). If 4, assume (N, C, H, W).
        if len(in_shape) == 4:
            c_in = in_shape[1]
        else:
            c_in = in_shape[0] if in_shape[0] <= 4 else 3 # Heuristic: if first dim is small, it's C
        
        n_classes = out_shape[0] if out_shape else 10

        # Architecture Design:
        # Stem: 3x3 Conv -> BN -> SiLU (Start with 32 channels to be efficient)
        # Stage 1: 2 FractalBlocks (32 channels, 32x32)
        # Downsample: AvgPool 2x2
        # Stage 2: 2 FractalBlocks (64 channels, 16x16)
        # Downsample: AvgPool 2x2
        # Stage 3: 3 FractalBlocks (128 channels, 8x8)
        # Downsample: AvgPool 2x2
        # Stage 4: 3 FractalBlocks (256 channels, 4x4)
        # Head: Global Avg Pool -> Linear
        
        # This avoids the 64-128-256-512 progression by starting at 32 and ending at 256.
        # It uses the "Fractal" idea of parallel branches from Model 1 but simplified for efficiency.
        # It uses SiLU (from Model 1).
        # It uses Depthwise Convs (from Model 1).
        
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 channels
        self.stage1 = nn.Sequential(
            FractalBlock(32),
            FractalBlock(32)
        )
        
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition 1: 32 -> 64
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 64 channels
        self.stage2 = nn.Sequential(
            FractalBlock(64),
            FractalBlock(64)
        )
        
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition 2: 64 -> 128
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 128 channels
        self.stage3 = nn.Sequential(
            FractalBlock(128),
            FractalBlock(128),
            FractalBlock(128)
        )
        
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition 3: 128 -> 256
        self.trans3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # Stage 4: 256 channels
        self.stage4 = nn.Sequential(
            FractalBlock(256),
            FractalBlock(256),
            FractalBlock(256)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)
        
        # Initialize weights
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
        x = self.pool1(x)
        x = self.trans1(x)
        
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.trans2(x)
        
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.trans3(x)
        
        x = self.stage4(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Model 1 (74.66%) used AdamW with lr=0.002, momentum=0.75
        # We use the same optimizer setup as the model to improve.
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