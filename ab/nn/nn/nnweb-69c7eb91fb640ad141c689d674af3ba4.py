import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        
        if use_residual and stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.act(self.bn1(self.depthwise(x)))
        out = self.act(self.bn2(self.pointwise(out)))
        return out + shortcut

class EfficientNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, stride=1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        mid_ch = in_ch * expansion
        
        # Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True)
        )
        
        # Depthwise
        self.depthwise = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, groups=mid_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_ch)
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        
        if use_residual and stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.expand(x)
        out = self.bn_dw(self.depthwise(out))
        out = self.project(out)
        return out + shortcut

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        c_in = 3 
        n_classes = out_shape[0] if out_shape else 10
        
        # Start with a wider stem to capture features early
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 -> 64, stride 1 (input 32x32 -> 32x32)
        self.stage1 = nn.Sequential(
            EfficientNetBlock(32, 64, expansion=1, stride=1),
            EfficientNetBlock(64, 64, expansion=4, stride=1),
        )
        
        # Stage 2: 64 -> 128, stride 2 (input 32x32 -> 16x16)
        self.stage2 = nn.Sequential(
            EfficientNetBlock(64, 128, expansion=4, stride=2),
            EfficientNetBlock(128, 128, expansion=4, stride=1),
            EfficientNetBlock(128, 128, expansion=4, stride=1),
        )
        
        # Stage 3: 128 -> 256, stride 2 (input 16x16 -> 8x8)
        self.stage3 = nn.Sequential(
            EfficientNetBlock(128, 256, expansion=4, stride=2),
            EfficientNetBlock(256, 256, expansion=4, stride=1),
            EfficientNetBlock(256, 256, expansion=4, stride=1),
        )
        
        # Stage 4: 256 -> 512, stride 2 (input 8x8 -> 4x4)
        self.stage4 = nn.Sequential(
            EfficientNetBlock(256, 512, expansion=4, stride=2),
            EfficientNetBlock(512, 512, expansion=4, stride=1),
        )
        
        # Global Pooling and Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, n_classes)
        
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
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