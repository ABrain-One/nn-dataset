import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = GELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=2):
        super().__init__()
        self.use_shortcut = (in_channels != out_channels) or (stride != 1)
        
        hidden_dim = in_channels * expansion
        
        # Expansion: 1x1 Conv -> BN -> GELU
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            GELU()
        )
        
        # Depthwise: 3x3 Depthwise Conv -> BN -> GELU
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            GELU()
        )
        
        # Projection: 1x1 Conv -> BN
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.expansion(x)
        out = self.depthwise(out)
        out = self.projection(out)
        
        out += identity
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        
        # Derive input channels
        if len(in_shape) == 3:
            c_in = in_shape[0]
        else:
            c_in = in_shape[1]
            
        n_classes = out_shape[0] if out_shape else 10

        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            GELU()
        )
        
        # Stage 1: 32 channels, 32x32
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1, expansion=2),
            ResidualBlock(32, 32, stride=1, expansion=2)
        )
        
        # Stage 2: 64 channels, 16x16
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, expansion=2),
            ResidualBlock(64, 64, stride=1, expansion=2)
        )
        
        # Stage 3: 128 channels, 8x8
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, expansion=2),
            ResidualBlock(128, 128, stride=1, expansion=2),
            ResidualBlock(128, 128, stride=1, expansion=2)
        )
        
        # Stage 4: 256 channels, 4x4
        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, expansion=2),
            ResidualBlock(256, 256, stride=1, expansion=2),
            ResidualBlock(256, 256, stride=1, expansion=2)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, n_classes)
        
        # Initialize weights
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
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
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