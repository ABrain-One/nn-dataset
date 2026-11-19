import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class FractalDWBlock(nn.Module):
    """
    A block that recursively processes the input.
    Depth 1: DW Conv -> BN -> SiLU -> 1x1 Conv -> BN
    Depth > 1: Parallel branches (Left/Right) processed recursively, combined via FractalDropPath.
    """
    def __init__(self, depth, channels, drop_prob=0.0):
        super().__init__()
        self.depth = depth
        self.channels = channels
        
        # Core operations
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(channels)
        
        if depth > 1:
            self.left = FractalDWBlock(depth - 1, channels, drop_prob)
            self.right_1 = FractalDWBlock(depth - 1, channels, drop_prob)
            self.right_2 = FractalDWBlock(depth - 1, channels, drop_prob)
            self.join = FractalDropPath(drop_prob)

    def forward(self, x):
        # Core processing
        y = self.dw_conv(x)
        y = self.bn_dw(y)
        y = self.act(y)
        y = self.pw_conv(y)
        y = self.bn_pw(y)
        
        if self.depth == 1:
            return y
        
        # Recursive processing
        out_left = self.left(x)
        out_right = self.right_2(self.right_1(x))
        
        return self.join([y, out_left, out_right])

class FractalDropPath(nn.Module):
    """
    Randomly drops paths during training, averages remaining paths.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        if not self.training:
            return torch.stack(inputs).mean(dim=0)
        
        n = len(inputs)
        mask = torch.bernoulli(torch.full((n,), 1 - self.drop_prob, device=inputs[0].device))
        if mask.sum() == 0:
            mask[torch.randint(0, n, (1,)).item()] = 1.0
        active = [inp for inp, m in zip(inputs, mask) if m > 0]
        return torch.stack(active).mean(dim=0)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.prm = prm
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 64, stride 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 64 channels, 32x32
        # Fractal depth 2
        self.stage1 = FractalDWBlock(depth=2, channels=64, drop_prob=0.0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 64 -> 128
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 128 channels, 16x16
        # Fractal depth 2
        self.stage2 = FractalDWBlock(depth=2, channels=128, drop_prob=0.0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Transition: 128 -> 256
        self.trans2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 256 channels, 8x8
        # Fractal depth 3 (deeper for higher resolution features)
        self.stage3 = FractalDWBlock(depth=3, channels=256, drop_prob=0.0)
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)
        
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
        # Normalize if needed
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
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with momentum for stability in 1 epoch
        # Model 1 (best) used AdamW, but SGD often generalizes better in single epoch if tuned right.
        # Let's try SGD with Nesterov-like behavior via momentum.
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
        
        self.batch_size = prm.get('batch', 128)
        return self.optimizer

    def learn(self, train_data):
        self.train()
        total_steps = len(train_data)
        
        # Cosine Annealing within 1 epoch
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