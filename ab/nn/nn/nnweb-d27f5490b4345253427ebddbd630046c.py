import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    """
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DepthwiseResBlock(nn.Module):
    """
    Lightweight Residual Block with Depthwise Separable Convolutions and SE Attention.
    Structure:
    1. 1x1 Conv (Expand) -> BN -> SiLU
    2. 3x3 Depthwise Conv -> BN -> SiLU
    3. 1x1 Conv (Project) -> BN
    4. SE Attention
    5. Residual Connection
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, use_se=True):
        super(DepthwiseResBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(round(in_channels * expand_ratio))
        
        # Expand
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_dim)
        
        # Project
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Attention
        self.se = SEBlock(out_channels, reduction=8) if use_se else nn.Identity()
        
        self.act = nn.SiLU(inplace=True)
        
        # Shortcut projection if needed
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        input = x
        
        # Expand
        y = self.act(self.expand_bn(self.expand_conv(x)))
        # Depthwise
        y = self.act(self.dw_bn(self.dw_conv(y)))
        # Project
        y = self.project_bn(self.project_conv(y))
        # Attention
        y = self.se(y)
        
        # Shortcut
        if not isinstance(self.shortcut, nn.Identity):
            input = self.shortcut(x)
            
        return input + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 48 with 5x5 kernel to capture context
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, 32x32 -> 32x32
        # 2 blocks
        self.stage1 = nn.Sequential(
            DepthwiseResBlock(48, 48, stride=1, expand_ratio=4, use_se=False),
            DepthwiseResBlock(48, 48, stride=1, expand_ratio=4, use_se=False)
        )
        
        # Transition 1: 48 -> 96, 32x32 -> 16x16
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 channels, 16x16 -> 16x16
        # 3 blocks
        self.stage2 = nn.Sequential(
            DepthwiseResBlock(96, 96, stride=1, expand_ratio=4, use_se=True),
            DepthwiseResBlock(96, 96, stride=1, expand_ratio=4, use_se=True),
            DepthwiseResBlock(96, 96, stride=1, expand_ratio=4, use_se=True)
        )
        
        # Transition 2: 96 -> 192, 16x16 -> 8x8
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 192 channels, 8x8 -> 8x8
        # 4 blocks
        self.stage3 = nn.Sequential(
            DepthwiseResBlock(192, 192, stride=1, expand_ratio=4, use_se=True),
            DepthwiseResBlock(192, 192, stride=1, expand_ratio=4, use_se=True),
            DepthwiseResBlock(192, 192, stride=1, expand_ratio=4, use_se=True),
            DepthwiseResBlock(192, 192, stride=1, expand_ratio=4, use_se=True)
        )
        
        # Transition 3: 192 -> 384, 8x8 -> 4x4
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 384 channels, 4x4 -> 4x4
        # 2 blocks
        self.stage4 = nn.Sequential(
            DepthwiseResBlock(384, 384, stride=1, expand_ratio=4, use_se=True),
            DepthwiseResBlock(384, 384, stride=1, expand_ratio=4, use_se=True)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(384, n_classes)
        
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
        # The harness typically applies the transform (norm_32_flip) to the data loader.
        # We assume inputs are already normalized and flipped if required by the harness.
        
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
        
        # Using AdamW as in the strongest model (Model 1)
        # Model 1 used lr=0.002, momentum=0.75 (beta1)
        # We will use a slightly higher LR to compensate for the different architecture depth, 
        # but keep it stable. 0.005 is a good middle ground for AdamW in 1-epoch.
        # Momentum is mapped to beta1.
        
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Estimate total steps for 50,000 CIFAR-10 images
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        # Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=prm['lr'] * 0.01
        )
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()