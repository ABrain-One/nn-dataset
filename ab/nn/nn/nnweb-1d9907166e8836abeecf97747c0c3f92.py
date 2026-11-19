import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x) * x
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio=6, use_cbam=False):
        super(InvertedResidual, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        
        # Expand
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # Project
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True)
        
        self.cbam = CBAM(hidden_dim, reduction=16) if use_cbam else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        
        x = self.cbam(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.use_residual:
            x = x + input
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        self.batch_size = prm.get('batch', 128)
        
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        
        # Reduced channel widths to stay under 5M parameters
        # Previous: 64 -> 128 -> 256 -> 512
        # New: 32 -> 64 -> 128 -> 256
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 -> 32
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 32, stride=1, expand_ratio=6, use_cbam=False),
            InvertedResidual(32, 32, stride=1, expand_ratio=6, use_cbam=True)
        )
        
        # Transition 1: 32 -> 64, stride 2
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 64 -> 64
        self.stage2 = nn.Sequential(
            InvertedResidual(64, 64, stride=1, expand_ratio=6, use_cbam=False),
            InvertedResidual(64, 64, stride=1, expand_ratio=6, use_cbam=True)
        )
        
        # Transition 2: 64 -> 128, stride 2
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 128 -> 128
        self.stage3 = nn.Sequential(
            InvertedResidual(128, 128, stride=1, expand_ratio=6, use_cbam=False),
            InvertedResidual(128, 128, stride=1, expand_ratio=6, use_cbam=True)
        )
        
        # Transition 3: 128 -> 256, stride 2
        self.trans3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 256 -> 256
        self.stage4 = nn.Sequential(
            InvertedResidual(256, 256, stride=1, expand_ratio=6, use_cbam=True)
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
        # No normalization in forward, trusting the harness or raw data robustness
        # as per the analysis of the strongest previous model.
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
        
        # Using AdamW with momentum mapped to beta1
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Estimate total steps for 50,000 CIFAR-10 images
        batch_size = prm.get('batch', 128)
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