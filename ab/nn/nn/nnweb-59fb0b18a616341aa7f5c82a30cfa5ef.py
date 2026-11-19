import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    """Squeeze and Excitation block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FractalDWBlock(nn.Module):
    """
    Recursive Depthwise Convolution Block with SE Attention.
    Base case: DW Conv -> BN -> SiLU -> 1x1 Conv -> BN -> SE
    Recursive case: Parallel branches (Left, Right) averaged, then passed through core.
    """
    def __init__(self, depth, channels, use_se=True):
        super().__init__()
        self.depth = depth
        self.channels = channels
        
        # Core processing
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # SE Block
        self.se = SEBlock(channels, reduction=16) if use_se else nn.Identity()
        
        if depth > 1:
            self.left = FractalDWBlock(depth - 1, channels, use_se=False) # Avoid nested SE to save params
            self.right = FractalDWBlock(depth - 1, channels, use_se=False)

    def forward(self, x):
        if self.depth > 1:
            left_out = self.left(x)
            right_out = self.right(x)
            # Average the parallel branches
            x = (left_out + right_out) / 2.0
        
        # Core
        y = self.dw_conv(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.pw_conv(y)
        y = self.bn2(y)
        y = self.se(y)
        
        # Residual connection
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels
        self.stage1 = nn.Sequential(
            FractalDWBlock(depth=2, channels=48, use_se=True),
            FractalDWBlock(depth=2, channels=48, use_se=True)
        )
        
        # Transition 1: 48 -> 96, Pool
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 channels
        self.stage2 = nn.Sequential(
            FractalDWBlock(depth=2, channels=96, use_se=True),
            FractalDWBlock(depth=2, channels=96, use_se=True),
            FractalDWBlock(depth=2, channels=96, use_se=True)
        )
        
        # Transition 2: 96 -> 192, Pool
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 192 channels
        self.stage3 = nn.Sequential(
            FractalDWBlock(depth=2, channels=192, use_se=True),
            FractalDWBlock(depth=2, channels=192, use_se=True),
            FractalDWBlock(depth=2, channels=192, use_se=True),
            FractalDWBlock(depth=2, channels=192, use_se=True)
        )
        
        # Transition 3: 192 -> 384, Pool
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 384 channels
        self.stage4 = nn.Sequential(
            FractalDWBlock(depth=2, channels=384, use_se=True),
            FractalDWBlock(depth=2, channels=384, use_se=True)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 100),
            nn.SiLU(inplace=True),
            nn.Linear(100, n_classes)
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
        # Assume harness handles normalization if 'transform' is set to norm_32.
        # If not, we might need to normalize here, but standard practice in these challenges
        # is that the DataLoader applies the transform.
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.trans3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with tuned hyperparameters
        # Model 1 used lr=0.002, momentum=0.75.
        # We use lr=0.003 and momentum=0.90 (beta1) for a slightly more aggressive start
        # with better stability.
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Estimate total steps
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        # Custom Scheduler: Linear Warmup + Cosine Decay
        # Warmup for 5% of steps
        warmup_steps = max(1, int(total_steps * 0.05))
        
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = prm['lr']
        
        return self.optimizer
    
    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
            
        # Recalculate warmup based on actual length if possible
        self.total_steps = total_steps
        self.warmup_steps = max(1, int(total_steps * 0.05))
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Update learning rate
            if i < self.warmup_steps:
                # Linear warmup
                lr = self.base_lr * (i + 1) / self.warmup_steps
            else:
                # Cosine decay
                progress = (i - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()