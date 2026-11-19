import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class AttentionBlock(nn.Module):
    """
    Lightweight attention block (SE-like) to provide global context.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        b, c, h, w = x.shape
        y = x.mean(dim=[2, 3])
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = torch.sigmoid(y)
        return x * y.view(b, c, 1, 1)

class FractalDWBlock(nn.Module):
    """
    Recursive depthwise convolution block.
    Base case: DW Conv -> BN -> SiLU -> 1x1 Conv -> BN
    Recursive case: Parallel branches (Left, Right) processed by smaller FractalDWBlocks,
    combined via averaging, then passed through core.
    """
    def __init__(self, depth, channels):
        super().__init__()
        self.depth = depth
        self.channels = channels
        
        # Core processing
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        if depth > 1:
            self.left = FractalDWBlock(depth - 1, channels)
            self.right = FractalDWBlock(depth - 1, channels)
            # Simple averaging for combination (FractalDropPath with prob 0)
            
    def forward(self, x):
        if self.depth > 1:
            left_out = self.left(x)
            right_out = self.right(x)
            # Average the outputs of the parallel branches
            x = (left_out + right_out) / 2.0
        
        # Core
        y = self.dw_conv(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.pw_conv(y)
        y = self.bn2(y)
        
        # Residual
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 64
        # Using a larger kernel to capture context early
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 64 channels, Depth 2 Fractal
        self.stage1 = FractalDWBlock(depth=2, channels=64)
        self.att1 = AttentionBlock(64)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 64 -> 128
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 128 channels, Depth 3 Fractal
        self.stage2 = FractalDWBlock(depth=3, channels=128)
        self.att2 = AttentionBlock(128)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 128 -> 256
        self.trans2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 256 channels, Depth 3 Fractal
        self.stage3 = FractalDWBlock(depth=3, channels=256)
        self.att3 = AttentionBlock(256)
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, n_classes)
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
        # Apply normalization if needed
        # CIFAR-10 mean/std
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.att1(x)
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.att2(x)
        x = self.pool2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.att3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with momentum as it is often more stable for 1-epoch training
        # when combined with a proper LR schedule.
        # Model 1 (74.66%) used AdamW. Let's try SGD with a cosine schedule.
        
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
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