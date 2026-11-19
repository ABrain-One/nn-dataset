import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 16))
        self.fc2 = nn.Linear(max(channels // reduction, 16), channels)

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y

class FractalUnit(nn.Module):
    """
    A depthwise-separable convolutional block with SE gating and optional recursive depth.
    """
    def __init__(self, channels, depth=1, drop_prob=0.0):
        super().__init__()
        self.depth = depth
        self.drop_prob = drop_prob
        
        # Depthwise Separable Conv: 3x3 DW + 1x1 PW
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(channels)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.se = SEBlock(channels)
        
        # Recursive branches if depth > 1
        if depth > 1:
            self.left = FractalUnit(channels, depth - 1, drop_prob)
            self.right = FractalUnit(channels, depth - 1, drop_prob)

    def forward(self, x):
        if self.depth == 1:
            identity = x
            x = self.dw_conv(x)
            x = self.bn_dw(x)
            x = self.act(x)
            x = self.pw_conv(x)
            x = self.bn_pw(x)
            x = self.act(x)
            x = self.se(x)
            return x + identity # Residual connection
        
        # Fractal combination
        out_left = self.left(x)
        out_right = self.right(x)
        
        # Combine via averaging (DropPath-like regularization)
        if self.training and self.drop_prob > 0:
            mask = torch.bernoulli(torch.full((2,), 1 - self.drop_prob, device=x.device))
            if mask.sum() == 0:
                mask[0] = 1.0
            active = []
            if mask[0] > 0: active.append(out_left)
            if mask[1] > 0: active.append(out_right)
            combined = torch.stack(active).mean(dim=0)
        else:
            combined = (out_left + out_right) / 2.0
            
        # Apply the base conv block to the combined result
        x = self.dw_conv(combined)
        x = self.bn_dw(x)
        x = self.act(x)
        x = self.pw_conv(x)
        x = self.bn_pw(x)
        x = self.act(x)
        x = self.se(x)
        
        return x + combined # Residual from combined

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        num_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 32 (32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32x32 -> 16x16, 32 -> 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            FractalUnit(64, depth=2, drop_prob=0.1)
        )
        
        # Stage 2: 16x16 -> 8x8, 64 -> 128
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            FractalUnit(128, depth=3, drop_prob=0.2)
        )
        
        # Stage 3: 8x8 -> 4x4, 128 -> 256
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            FractalUnit(256, depth=2, drop_prob=0.3)
        )
        
        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
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
        # Apply normalization if specified in transform
        if self.transform == 'norm_32':
            x = (x - 0.4914) / 0.2470
        elif self.transform == 'norm_32_flip':
            x = (x - 0.4914) / 0.2470
            
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        # AdamW is crucial for 1-epoch training
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=prm['lr'], 
            betas=(prm['momentum'], 0.999), 
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Estimate total steps for scheduler
        # CIFAR-10 has 50,000 images.
        # We assume standard batch size from prm.
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        # Cosine annealing to 0
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps, 
            eta_min=0.0
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