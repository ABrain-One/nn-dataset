import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class FractalBlock(nn.Module):
    """
    Recursive depthwise block.
    Base case: DW Conv -> BN -> GELU -> PW Conv -> BN
    Recursive case: Parallel branches (Left, Right) processed by smaller FractalDWBlocks,
    combined via averaging, then passed through core.
    """
    def __init__(self, depth, channels):
        super().__init__()
        self.depth = depth
        self.channels = channels
        
        # Core processing
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        if depth > 1:
            self.left = FractalBlock(depth - 1, channels)
            self.right = FractalBlock(depth - 1, channels)

    def forward(self, x):
        if self.depth > 1:
            left_out = self.left(x)
            right_out = self.right(x)
            x = (left_out + right_out) / 2.0
        
        # Core
        y = self.dw_conv(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.pw_conv(y)
        y = self.bn2(y)
        
        # Residual
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 48
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, 48, kernel_size=3, padding=1),
        )
        
        # Stage 1: 48 channels, Depth 2 Fractal
        self.stage1 = FractalBlock(depth=2, channels=48)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 48 -> 96
        self.trans1 = nn.Sequential(
            ConvBNAct(48, 96, kernel_size=1, padding=0),
        )
        
        # Stage 2: 96 channels, Depth 2 Fractal
        self.stage2 = FractalBlock(depth=2, channels=96)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 96 -> 192
        self.trans2 = nn.Sequential(
            ConvBNAct(96, 192, kernel_size=1, padding=0),
        )
        
        # Stage 3: 192 channels, Depth 3 Fractal
        self.stage3 = FractalBlock(depth=3, channels=192)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 192 -> 384
        self.trans3 = nn.Sequential(
            ConvBNAct(192, 384, kernel_size=1, padding=0),
        )
        
        # Stage 4: 384 channels, Depth 2 Fractal
        self.stage4 = FractalBlock(depth=2, channels=384)
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, n_classes)
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
        # CIFAR-10 mean/std
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.trans3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
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