import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class FractalUnit(nn.Module):
    """
    A block that processes the input through a main path and a parallel 'ghost' path,
    then combines them. This adds depth and non-linearity without strictly increasing
    the parameter count linearly as much as a deep serial stack would for the same
    receptive field, and encourages diverse feature extraction.
    """
    def __init__(self, channels, depth=2):
        super().__init__()
        # Main path: Depthwise -> Pointwise
        self.main_dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.main_bn = nn.BatchNorm2d(channels)
        self.main_pw = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.main_bn2 = nn.BatchNorm2d(channels)
        
        # Ghost path: Simple 1x1 with fewer channels (or same) to create a shortcut-like feature
        # To keep params low, we can use a 1x1 conv that projects to half channels then back?
        # Or just a simple 1x1. Let's do a 1x1 conv that acts as a feature mixer.
        self.ghost_pw = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.ghost_bn = nn.BatchNorm2d(channels)
        
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # Main path
        y = self.main_dw(x)
        y = self.main_bn(y)
        y = self.main_pw(y)
        y = self.main_bn2(y)
        
        # Ghost path
        g = self.ghost_pw(x)
        g = self.ghost_bn(g)
        
        # Combine
        out = y + g
        out = self.act(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32')
        self.batch_size = prm.get('batch', 128)
        
        in_ch = in_shape[1]
        self.n_classes = out_shape[0] if out_shape else 10
        
        # Non-standard channel progression: 48 -> 96 -> 192 -> 384
        # Avoiding 64-128-256-512
        
        # Stem: Wide kernel for better initial context
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 16x16 -> 16x16 (Resolution maintained, channels constant)
        # Using FractalUnits for depth
        self.stage1 = nn.Sequential(
            FractalUnit(48),
            FractalUnit(48),
            FractalUnit(48)
        )
        
        # Transition 1: 16x16 -> 8x8, 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 8x8 -> 8x8
        self.stage2 = nn.Sequential(
            FractalUnit(96),
            FractalUnit(96),
            FractalUnit(96)
        )
        
        # Transition 2: 8x8 -> 4x4, 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 4x4 -> 4x4
        self.stage3 = nn.Sequential(
            FractalUnit(192),
            FractalUnit(192)
        )
        
        # Transition 3: 4x4 -> 2x2, 192 -> 384
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        # Stage 4: 2x2 -> 2x2
        self.stage4 = nn.Sequential(
            FractalUnit(384)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(384, 1024),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, self.n_classes)
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
        # Apply normalization manually to ensure it's applied even if the dataloader didn't do it perfectly
        # CIFAR-10 mean/std
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(x.device)
        
        if self.transform in ['norm_32', 'norm_32_flip']:
            x = (x - mean) / std
            
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.trans3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # SGD with momentum
        # Learning rate is critical. For 1 epoch, we need a high start and decay.
        # We will use Cosine Annealing.
        
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
        
        # We need total steps. CIFAR-10 has 50k images.
        # batch size is in prm.
        # If batch size is 128, steps = 390.
        # If batch size is 64, steps = 781.
        
        # We will estimate total steps based on standard CIFAR-10 size (50,000)
        # However, it's safer to compute it in learn() if possible, but scheduler needs it at init.
        # Let's assume standard CIFAR-10 size.
        # If the dataset is different, this might be slightly off, but cosine is robust.
        
        # Actually, let's try to be smarter. We can't access dataset size in train_setup easily if it's not passed.
        # But `prm` has 'batch'.
        # Let's assume 50,000 samples.
        n_samples = 50000
        total_steps = n_samples // prm['batch']
        
        # Add a small buffer in case of drop_last or different dataset size
        # CosineAnnealingLR with T_max = total_steps
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps, 
            eta_min=0.0
        )
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        # To be robust, let's try to get the actual number of batches if possible
        # and re-create the scheduler if the estimate was way off?
        # No, simpler: just step the scheduler.
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Ensure labels are 1D long tensor for CrossEntropyLoss
            if labels.dim() > 1:
                labels = labels.squeeze()
            labels = labels.long()
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()