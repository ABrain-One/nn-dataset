import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class MultiScaleBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(MultiScaleBlock, self).__init__()
        mid = max(1, channels // reduction)
        
        # Branch 1: Standard 3x3
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Branch 2: Depthwise 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Branch 3: 1x1 (Global context per pixel)
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Attention Gate: Learn how to combine these branches
        # Outputs 3 scalar weights (one per branch)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, 3),
            nn.Softmax(dim=1)
        )
        
        self.bn_out = nn.BatchNorm2d(channels)
        self.act_out = nn.SiLU(inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Compute gate weights: (B, 3)
        gate_weights = self.gate(x) # (B, 3)
        
        # Expand to (B, 3, 1, 1) to broadcast over (B, C, H, W)
        gate_weights = gate_weights.view(gate_weights.size(0), 3, 1, 1)
        
        # Weighted sum: (B, 3, C, H, W) * (B, 3, 1, 1) -> sum over dim 1
        # b1 is (B, C, H, W). unsqueeze(1) -> (B, 1, C, H, W).
        # gate_weights[:, 0:1, :, :] is (B, 1, 1, 1).
        # Product is (B, 1, C, H, W).
        # Summing the three terms gives (B, C, H, W).
        
        out = (b1.unsqueeze(1) * gate_weights[:, 0:1, :, :]) + \
              (b2.unsqueeze(1) * gate_weights[:, 1:2, :, :]) + \
              (b3.unsqueeze(1) * gate_weights[:, 2:3, :, :])
        
        # Sum over the branch dimension (dim 1) to get (B, C, H, W)
        out = out.sum(dim=1)
        
        out = self.bn_out(out)
        out = self.act_out(out)
        
        # Residual
        return out + x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        self.batch_size = prm.get('batch', 128)
        
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
            MultiScaleBlock(48),
            MultiScaleBlock(48)
        )
        
        # Transition 1: 48 -> 96, Pool 2x
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 channels
        self.stage2 = nn.Sequential(
            MultiScaleBlock(96),
            MultiScaleBlock(96),
            MultiScaleBlock(96)
        )
        
        # Transition 2: 96 -> 192, Pool 2x
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 192 channels
        self.stage3 = nn.Sequential(
            MultiScaleBlock(192),
            MultiScaleBlock(192)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, n_classes)
        
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
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
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
        
        # Use SGD with momentum for stability in 1 epoch
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4,
            nesterov=True
        )
        
        # Cosine Annealing Schedule
        # Estimate steps
        self.total_steps = 50000 // prm['batch']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps,
            eta_min=prm['lr'] * 0.01
        )
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()