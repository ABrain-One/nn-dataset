import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return ['lr', 'momentum', 'batch', 'transform']

class SEBlock(nn.Module):
    """Squeeze and Excitation block for channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y

class DropPath(nn.Module):
    """DropPath (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class EfficientBlock(nn.Module):
    """
    Depthwise Separable Convolution Block with SE and DropPath.
    Structure:
    1. Expand (Pointwise 1x1)
    2. Depthwise 3x3
    3. Squeeze-Excite
    4. Project (Pointwise 1x1)
    5. Residual Connection (with DropPath)
    """
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=2, drop_prob=0.0):
        super().__init__()
        expanded = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch

        # Expansion
        self.conv1 = nn.Conv2d(in_ch, expanded, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded)

        # Depthwise
        self.conv2 = nn.Conv2d(expanded, expanded, 3, stride=stride, padding=1, groups=expanded, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded)

        # SE
        self.se = SEBlock(expanded)

        # Projection
        self.conv3 = nn.Conv2d(expanded, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.drop_path = DropPath(drop_prob)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = self.bn3(self.conv3(x))
        
        if self.use_residual:
            x = self.drop_path(x) + input
        else:
            x = self.drop_path(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        num_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 32 -> 32 (2 blocks), then 32 -> 64 (1 block with stride 2)
        # Input 32x32 -> Output 16x16
        self.stage1 = nn.Sequential(
            EfficientBlock(32, 32, stride=1, expand_ratio=2, drop_prob=0.05),
            EfficientBlock(32, 32, stride=1, expand_ratio=2, drop_prob=0.05),
            EfficientBlock(32, 64, stride=2, expand_ratio=2, drop_prob=0.10)
        )

        # Stage 2: 64 -> 64 (2 blocks), then 64 -> 128 (1 block with stride 2)
        # Input 16x16 -> Output 8x8
        self.stage2 = nn.Sequential(
            EfficientBlock(64, 64, stride=1, expand_ratio=2, drop_prob=0.15),
            EfficientBlock(64, 64, stride=1, expand_ratio=2, drop_prob=0.15),
            EfficientBlock(64, 128, stride=2, expand_ratio=2, drop_prob=0.20)
        )

        # Stage 3: 128 -> 128 (2 blocks), then 128 -> 256 (1 block with stride 2)
        # Input 8x8 -> Output 4x4
        self.stage3 = nn.Sequential(
            EfficientBlock(128, 128, stride=1, expand_ratio=2, drop_prob=0.25),
            EfficientBlock(128, 128, stride=1, expand_ratio=2, drop_prob=0.25),
            EfficientBlock(128, 256, stride=2, expand_ratio=2, drop_prob=0.30)
        )

        # Stage 4: 256 -> 256 (2 blocks), then 256 -> 512 (1 block with stride 2)
        # Input 4x4 -> Output 2x2
        self.stage4 = nn.Sequential(
            EfficientBlock(256, 256, stride=1, expand_ratio=2, drop_prob=0.35),
            EfficientBlock(256, 256, stride=1, expand_ratio=2, drop_prob=0.35),
            EfficientBlock(256, 512, stride=2, expand_ratio=2, drop_prob=0.40)
        )

        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
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
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        
        # AdamW is chosen based on Model 1's success.
        # momentum is used as beta1.
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=prm['lr'], 
            betas=(prm['momentum'], 0.999), 
            weight_decay=1e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # We need to set up the scheduler. 
        # We don't know the exact number of batches in train_setup reliably 
        # without the dataset length, but we can estimate or set it in learn.
        # For now, we store the necessary info.
        self.batch_size = prm['batch']
        self.base_lr = prm['lr']
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        # Estimate total steps for scheduling
        # CIFAR-10 has 50,000 images.
        # If we don't know the exact length, we assume standard CIFAR-10.
        # However, train_data is a DataLoader. We can try to get its length.
        try:
            total_steps = len(train_data)
        except TypeError:
            total_steps = 50000 // self.batch_size  # Fallback for CIFAR-10
            
        # Cosine Annealing with Warmup
        # Warmup for first 10% of steps, then cosine decay.
        warmup_steps = int(total_steps * 0.1)
        
        # We implement a manual scheduler to have full control within the 1 epoch.
        # Or use CosineAnnealingLR with a warmup wrapper.
        # Let's use a simple linear warmup + cosine decay manually to be safe.
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Learning Rate Schedule
            if i < warmup_steps:
                # Linear warmup
                lr = self.base_lr * (i + 1) / warmup_steps
            else:
                # Cosine decay
                progress = (i - warmup_steps) / (total_steps - warmup_steps)
                lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            
            # Update LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()