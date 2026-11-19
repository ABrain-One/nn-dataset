import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class ConvBlock(nn.Module):
    """Standard Conv-BN-Act block with optional depthwise separation."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, use_dw=False, act=nn.GELU):
        super().__init__()
        if use_dw:
            self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size, stride, kernel_size // 2, groups=in_ch, bias=False)
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.conv2 = nn.Conv2d(in_ch, out_ch, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride, kernel_size // 2, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = None
            self.bn2 = None
        self.act = act()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.conv2 is not None:
            x = self.conv2(x)
            x = self.bn2(x)
        return self.act(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.mean(dim=(2, 3))
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.act(y)
        return x * y.view(b, c, 1, 1)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 32 (smaller start to save params)
        self.stem = ConvBlock(in_ch, 32, kernel_size=3, stride=1)
        
        # Stage 1: 32 channels, spatial size /2
        # Uses depthwise convs for efficiency
        self.stage1 = nn.Sequential(
            ConvBlock(32, 32, kernel_size=3, stride=2, use_dw=True),
            ConvBlock(32, 32, kernel_size=3, stride=1, use_dw=True),
            SEBlock(32, reduction=4)
        )
        
        # Stage 2: 64 channels, spatial size /4
        self.stage2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2, use_dw=True),
            ConvBlock(64, 64, kernel_size=3, stride=1, use_dw=True),
            ConvBlock(64, 64, kernel_size=3, stride=1, use_dw=True),
            SEBlock(64, reduction=4)
        )
        
        # Stage 3: 128 channels, spatial size /8
        self.stage3 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, use_dw=True),
            ConvBlock(128, 128, kernel_size=3, stride=1, use_dw=True),
            ConvBlock(128, 128, kernel_size=3, stride=1, use_dw=True),
            SEBlock(128, reduction=4)
        )
        
        # Stage 4: 256 channels, spatial size /16 (optional, depends on input size)
        # CIFAR is 32x32, so after 3 downsamples we are at 4x4.
        # Let's keep it at 3 stages to maintain resolution for classification.
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, n_classes)
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
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with momentum as provided
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=5e-4
        )
        
        self.batch_size = prm.get('batch', 128)
        
        return self.optimizer
    
    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
        
        # Cosine schedule within 1 epoch
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            scheduler.step()