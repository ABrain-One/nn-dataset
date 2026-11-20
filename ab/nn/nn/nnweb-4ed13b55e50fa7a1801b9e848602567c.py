import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class MultiScaleBlock(nn.Module):
    """
    A block that combines a standard path with a dilated path and a global context path (SE-like).
    Uses SiLU activation and BatchNorm.
    """
    def __init__(self, in_channels, out_channels, stride=1, se_reduction=8):
        super().__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        
        # Path 1: Standard 3x3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Path 2: Dilated 3x3 (receptive field expansion)
        # Only active if stride == 1 to maintain spatial dimensions easily.
        self.has_dilated = (stride == 1)
        if self.has_dilated:
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Path 3: Global Context (SE style)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, max(1, out_channels // se_reduction)),
            nn.SiLU(inplace=True),
            nn.Linear(max(1, out_channels // se_reduction), out_channels),
            nn.Sigmoid()
        )
        
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        # Path 1
        y1 = self.act(self.bn1(self.conv1(x)))
        
        # Path 2
        if self.has_dilated:
            y2 = self.act(self.bn2(self.conv2(x)))
            y = y1 + y2
        else:
            y = y1
            
        # Apply SE to the fused feature
        b, c, h, w = y.size()
        se_weight = self.se(y) # (B, C)
        se_weight = se_weight.view(b, c, 1, 1)
        y = y * se_weight
        
        if self.use_residual:
            y = y + x
            
        return y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        num_classes = out_shape[0]
        
        # Reduced channel widths: 24 -> 48 -> 96 -> 192
        # This keeps the "doubling" progression but starts lower to save parameters.
        
        # Stem: 3 -> 24
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 24 -> 24 (2 blocks), stride 1
        self.stage1 = nn.Sequential(
            MultiScaleBlock(24, 24, stride=1),
            MultiScaleBlock(24, 24, stride=1)
        )
        
        # Transition 1: 24 -> 48, stride 2
        self.trans1 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 48 -> 48 (3 blocks), stride 1
        self.stage2 = nn.Sequential(
            MultiScaleBlock(48, 48, stride=1),
            MultiScaleBlock(48, 48, stride=1),
            MultiScaleBlock(48, 48, stride=1)
        )
        
        # Transition 2: 48 -> 96, stride 2
        self.trans2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 96 -> 96 (4 blocks), stride 1
        self.stage3 = nn.Sequential(
            MultiScaleBlock(96, 96, stride=1),
            MultiScaleBlock(96, 96, stride=1),
            MultiScaleBlock(96, 96, stride=1),
            MultiScaleBlock(96, 96, stride=1)
        )
        
        # Transition 3: 96 -> 192, stride 2
        self.trans3 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 192 -> 192 (2 blocks), stride 1
        self.stage4 = nn.Sequential(
            MultiScaleBlock(192, 192, stride=1),
            MultiScaleBlock(192, 192, stride=1)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, num_classes)
        
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
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use AdamW, as it was effective in Model 1.
        # Map 'momentum' to beta1 for AdamW.
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=prm['lr'], 
            betas=(prm['momentum'], 0.999), 
            weight_decay=1e-4
        )
        
        # Estimate total steps for scheduler
        batch_size = prm.get('batch', 128)
        # CIFAR-10 has 50,000 images.
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()