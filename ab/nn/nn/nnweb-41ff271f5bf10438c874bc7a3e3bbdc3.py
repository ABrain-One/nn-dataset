import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class EfficientBlock(nn.Module):
    """
    EfficientNet-style block with SE attention and residual connection.
    Uses depthwise separable convolutions for parameter efficiency.
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(round(in_channels * expand_ratio))
        
        # Expand
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_dim)
        
        # Squeeze-and-Excite
        se_channels = max(1, int(hidden_dim * se_ratio))
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(hidden_dim, se_channels, kernel_size=1, bias=False)
        self.se_fc2 = nn.Conv2d(se_channels, hidden_dim, kernel_size=1, bias=False)
        
        # Project
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        input = x
        
        # Expand
        y = self.expand_conv(x)
        y = self.expand_bn(y)
        y = self.act(y)
        
        # Depthwise
        y = self.dw_conv(y)
        y = self.dw_bn(y)
        y = self.act(y)
        
        # SE
        se = self.se_pool(y)
        se = self.se_fc1(se)
        se = self.act(se)
        se = self.se_fc2(se)
        se = self.act(se)
        y = y * se
        
        # Project
        y = self.project_conv(y)
        y = self.project_bn(y)
        
        if self.use_residual:
            y = y + input
        
        return y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        self.batch_size = prm.get('batch', 128)
        
        in_channels = in_shape[1]
        num_classes = out_shape[0] if out_shape else 10
        
        # Reduced channel counts to stay under 5M parameters
        # Stem: 3 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 -> 32, 32x32
        self.stage1 = nn.Sequential(
            EfficientBlock(32, 32, stride=1, expand_ratio=4, se_ratio=0.25)
        )
        
        # Transition: 32 -> 64, 32 -> 16
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 64 -> 64, 16x16
        self.stage2 = nn.Sequential(
            EfficientBlock(64, 64, stride=1, expand_ratio=4, se_ratio=0.25),
            EfficientBlock(64, 64, stride=1, expand_ratio=4, se_ratio=0.25)
        )
        
        # Transition: 64 -> 128, 16 -> 8
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 128 -> 128, 8x8
        self.stage3 = nn.Sequential(
            EfficientBlock(128, 128, stride=1, expand_ratio=4, se_ratio=0.25),
            EfficientBlock(128, 128, stride=1, expand_ratio=4, se_ratio=0.25),
            EfficientBlock(128, 128, stride=1, expand_ratio=4, se_ratio=0.25)
        )
        
        # Transition: 128 -> 256, 8 -> 4
        self.trans3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 256 -> 256, 4x4
        self.stage4 = nn.Sequential(
            EfficientBlock(256, 256, stride=1, expand_ratio=4, se_ratio=0.25),
            EfficientBlock(256, 256, stride=1, expand_ratio=4, se_ratio=0.25)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
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
        # Apply normalization if needed
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
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
        
        # SGD with momentum for stability in 1 epoch
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
        
        # Store batch size for scheduling
        self.batch_size = prm['batch']
        
        return self.optimizer
    
    def learn(self, train_data):
        self.train()
        total_steps = len(train_data)
        
        # Cosine annealing within 1 epoch
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