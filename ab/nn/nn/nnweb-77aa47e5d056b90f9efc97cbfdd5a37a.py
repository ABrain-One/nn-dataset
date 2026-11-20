import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'transform'}

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, activation='silu'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, kernel_size // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.act(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class EfficientBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=4, se_reduction=16):
        super().__init__()
        mid_ch = in_ch * expansion
        
        # 1x1 expansion
        self.expand = ConvBNAct(in_ch, mid_ch, 1, 1, 1, 'silu')
        # 3x3 depthwise
        self.depthwise = ConvBNAct(mid_ch, mid_ch, 3, stride, mid_ch, 'silu')
        # SE block
        self.se = SEBlock(mid_ch, se_reduction)
        # 1x1 projection
        self.project = ConvBNAct(mid_ch, out_ch, 1, 1, 1, 'silu')

        # Shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        out += identity
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32')
        
        in_ch = in_shape[1]
        out_ch = out_shape[0]
        h, w = in_shape[2], in_shape[3]
        
        # Stem
        # Use a slightly wider stem to capture features early
        self.stem = ConvBNAct(in_ch, 48, 3, 1, 1, 'silu')
        
        # Feature Pyramid
        # Stage 1: 48 -> 48 (no downsample)
        self.stage1 = EfficientBlock(48, 48, stride=1, expansion=4, se_reduction=16)
        
        # Stage 2: 48 -> 96 (downsample)
        self.stage2 = EfficientBlock(48, 96, stride=2, expansion=4, se_reduction=16)
        
        # Stage 3: 96 -> 192 (downsample)
        self.stage3 = EfficientBlock(96, 192, stride=2, expansion=4, se_reduction=16)
        
        # Stage 4: 192 -> 384 (downsample) - High capacity
        self.stage4 = EfficientBlock(192, 384, stride=2, expansion=4, se_reduction=16)
        
        # Final Classifier
        # We use AdaptiveAvgPool to handle any input size robustly
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(384, out_ch)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        
        # Use AdamW for faster convergence in 1 epoch
        # Momentum is mapped to beta1 for AdamW
        beta1 = prm.get('momentum', 0.9)
        # Ensure beta1 is in valid range for Adam
        if beta1 >= 1.0: beta1 = 0.99
        if beta1 < 0.0: beta1 = 0.0
        
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=prm['lr'], 
            betas=(beta1, 0.999), 
            weight_decay=1e-4
        )
        
        # Cosine Annealing for smooth convergence
        # Total steps is determined in learn()
        # We set a placeholder here, actual scheduler is created in learn
        self.criterion = nn.CrossEntropyLoss()
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        # Estimate total steps for scheduler
        # We don't know the exact length of the loader immediately if it's a generator, 
        # but typically it's a DataLoader. We can peek or just use a large number if unknown.
        # However, standard practice is to know the length.
        # Let's try to get length.
        try:
            total_steps = len(train_data)
        except TypeError:
            # If it's a generator, we don't know. We'll just run without scheduler or use a fixed count.
            # For CIFAR-10, it's 50000 / batch_size.
            # Let's assume standard CIFAR-10 size if length fails, or just run.
            total_steps = 50000 // 128 # Fallback guess
            
        # Create scheduler based on total steps
        # Cosine annealing from initial LR to 0
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=0)
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()