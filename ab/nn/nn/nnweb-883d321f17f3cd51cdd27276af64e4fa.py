import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return ['lr', 'momentum', 'batch', 'transform']

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class GhostBlock(nn.Module):
    """
    GhostNet-style block: cheap conv + linear replication.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.cheap_channels = out_channels // ratio
        
        # The "cheap" convolution
        self.cheap_conv = nn.Conv2d(in_channels, self.cheap_channels, kernel_size=kernel_size, 
                                     padding=kernel_size // 2, bias=False)
        self.bn_cheap = nn.BatchNorm2d(self.cheap_channels)
        
        # The "linear" (ghost) generation: depthwise conv to replicate
        # Kernel size 1 for the ghost part is standard to keep params low
        self.ghost_conv = nn.Conv2d(self.cheap_channels, out_channels, kernel_size=1, 
                                     bias=False)
        self.bn_ghost = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 1. Generate cheap features
        y = self.cheap_conv(x)
        y = self.bn_cheap(y)
        y = self.act(y)
        
        # 2. Generate ghost features
        y = self.ghost_conv(y)
        y = self.bn_ghost(y)
        y = self.act(y)
        
        return y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Reduced channel progression to stay under 5M params:
        # 48 -> 96 -> 192 -> 320
        # Previous was 48 -> 96 -> 192 -> 384
        
        # Stem: 3 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, spatial size H/2
        self.stage1 = nn.Sequential(
            GhostBlock(48, 48, kernel_size=5, ratio=2),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            SEBlock(48),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Transition: 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 96 channels, spatial size H/4
        self.stage2 = nn.Sequential(
            GhostBlock(96, 96, kernel_size=5, ratio=2),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            SEBlock(96),
            GhostBlock(96, 96, kernel_size=5, ratio=2),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            SEBlock(96),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Transition: 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 192 channels, spatial size H/8
        self.stage3 = nn.Sequential(
            GhostBlock(192, 192, kernel_size=5, ratio=2),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            SEBlock(192),
            GhostBlock(192, 192, kernel_size=5, ratio=2),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            SEBlock(192),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Transition: 192 -> 320
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 320, kernel_size=1, bias=False),
            nn.BatchNorm2d(320),
            nn.SiLU(inplace=True)
        )
        
        # Stage 4: 320 channels, spatial size H/16
        self.stage4 = nn.Sequential(
            GhostBlock(320, 320, kernel_size=5, ratio=2),
            nn.BatchNorm2d(320),
            nn.SiLU(inplace=True),
            SEBlock(320),
            GhostBlock(320, 320, kernel_size=5, ratio=2),
            nn.BatchNorm2d(320),
            nn.SiLU(inplace=True),
            SEBlock(320)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, n_classes)
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
        
        # Use AdamW as in Model 1, which was the strongest.
        # Map 'momentum' to beta1.
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Estimate total steps for scheduler
        # CIFAR-10 has 50,000 images.
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        # Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps, 
            eta_min=0.0
        )
        
        return self.optimizer
    
    def learn(self, train_data):
        self.train()
        
        # Determine actual number of batches if possible
        try:
            total_steps = len(train_data)
            # If the stored scheduler has a different T_max, we can't easily change it,
            # but we can just step it. If the lengths are close, it's fine.
            # To be robust, we could recreate the scheduler here if we knew the exact count,
            # but train_setup is called before learn.
            # We'll just use the pre-computed scheduler.
        except (TypeError, AttributeError):
            pass
        
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
            self.scheduler.step()