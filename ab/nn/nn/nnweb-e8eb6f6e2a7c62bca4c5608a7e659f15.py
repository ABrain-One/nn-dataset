import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class FractalConvBlock(nn.Module):
    """
    A recursive block inspired by Model 1's FractalBlock.
    It creates a binary tree of depth 'depth'.
    Leaves are depthwise-separable convolutions.
    Internal nodes merge branches via a weighted average (FractalDropPath logic).
    """
    def __init__(self, depth, channels, drop_prob=0.0):
        super().__init__()
        self.depth = depth
        self.channels = channels
        self.drop_prob = drop_prob
        
        if depth == 1:
            # Leaf: Depthwise Conv -> BN -> SiLU -> 1x1 Conv -> BN
            self.conv_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
            self.bn_dw = nn.BatchNorm2d(channels)
            self.act = nn.SiLU(inplace=True)
            self.conv_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.bn_pw = nn.BatchNorm2d(channels)
        else:
            # Internal Node: Two recursive children
            self.left = FractalConvBlock(depth - 1, channels, drop_prob)
            self.right = FractalConvBlock(depth - 1, channels, drop_prob)
            # Merging mechanism: Simple weighted average based on random mask (stochastic depth variant)
            # To keep it deterministic and efficient, we use a learned gate or just average.
            # Model 1 used stochastic drop. For stability in 1 epoch, we use a learned gate or simple mean.
            # Let's use a simple learned gate (1x1 conv to 2 channels, softmax) to blend left/right.
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 2),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        if self.depth == 1:
            y = self.conv_dw(x)
            y = self.bn_dw(y)
            y = self.act(y)
            y = self.conv_pw(y)
            y = self.bn_pw(y)
            # Residual connection at the leaf level to help gradient flow
            return x + y

        # Internal node
        out_left = self.left(x)
        out_right = self.right(x)
        
        # Compute gate
        gate = self.gate(x) # Shape: (B, 2)
        w_left = gate[:, 0:1] # (B, 1)
        w_right = gate[:, 1:2] # (B, 1)
        
        # Weighted sum
        out = w_left.unsqueeze(-1).unsqueeze(-1) * out_left + w_right.unsqueeze(-1).unsqueeze(-1) * out_right
        
        # Residual connection at the node level
        return x + out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.prm = prm
        
        in_channels = in_shape[1]
        num_classes = out_shape[0] if out_shape else 10
        
        # Channel progression: 48 -> 96 -> 192 -> 384
        # This avoids the standard 64-128-256-512 and keeps param count low.
        
        # Stem: 3 -> 48, stride 1 (keep 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, 32x32
        # Depth 2 Fractal Block
        self.stage1 = FractalConvBlock(depth=2, channels=48)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # 32 -> 16
        
        # Transition: 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 96 channels, 16x16
        # Depth 3 Fractal Block (more capacity)
        self.stage2 = FractalConvBlock(depth=3, channels=96)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # 16 -> 8
        
        # Transition: 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 192 channels, 8x8
        # Depth 3 Fractal Block
        self.stage3 = FractalConvBlock(depth=3, channels=192)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2) # 8 -> 4
        
        # Transition: 192 -> 384
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        # Stage 4: 384 channels, 4x4
        # Depth 2 Fractal Block (shallow spatial, deep channel)
        self.stage4 = FractalConvBlock(depth=2, channels=384)
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(384, num_classes)
        
        self._init_weights()
        self.to(device)

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
        # Normalize input if transform suggests it
        # CIFAR-10 mean/std
        if self.prm.get('transform', 'norm_32') in ['norm_32', 'norm_32_flip']:
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(x.device)
            x = (x - mean) / std

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
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use AdamW for faster convergence in 1 epoch
        # Map momentum to beta1
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Cosine Annealing for 1 epoch
        # Estimate total steps
        # CIFAR-10 has 50,000 training images
        # Assume batch size is provided in prm
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=prm['lr'] * 0.01
        )
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()