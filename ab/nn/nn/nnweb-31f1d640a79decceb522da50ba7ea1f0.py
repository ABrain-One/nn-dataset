import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class EfficientBlock(nn.Module):
    def __init__(self, channels, expand_ratio=4, drop_prob=0.0):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.drop_prob = drop_prob
        
        # Inverted Bottleneck: 1x1 expansion -> 3x3 depthwise -> 1x1 projection
        hidden_dim = int(channels * expand_ratio)
        
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.conv3 = nn.Conv2d(hidden_dim, channels, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.act = nn.SiLU(inplace=True)
        
        # Stochastic Depth (DropPath)
        self.drop_path = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        
        y = self.conv3(y)
        y = self.bn3(y)
        
        y = self.drop_path(y)
        
        return x + y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        
        c_in = in_shape[1]
        n_classes = out_shape[0]
        
        # Architecture Configuration
        # Avoiding 64-128-256-512 progression.
        # Using a progression that keeps spatial resolution longer and uses efficient channels.
        # Stages: [channels, num_blocks]
        # Input 32x32 -> 16x16 -> 8x8 -> 4x4
        # Channels: 32 -> 64 -> 128 -> 256
        # This is different from Model 1 (64->128->256) and Model 4 (64->128->256->512).
        
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 channels, 16x16 spatial
        # 2 blocks
        self.stage1 = nn.Sequential(
            EfficientBlock(32, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(32, expand_ratio=4, drop_prob=0.0)
        )
        
        # Transition 1: 32 -> 64 channels, 16x16 -> 8x8
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 64 channels, 8x8 spatial
        # 3 blocks
        self.stage2 = nn.Sequential(
            EfficientBlock(64, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(64, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(64, expand_ratio=4, drop_prob=0.0)
        )
        
        # Transition 2: 64 -> 128 channels, 8x8 -> 4x4
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 128 channels, 4x4 spatial
        # 4 blocks
        self.stage3 = nn.Sequential(
            EfficientBlock(128, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(128, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(128, expand_ratio=4, drop_prob=0.0),
            EfficientBlock(128, expand_ratio=4, drop_prob=0.0)
        )
        
        # Final Head
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with a small hidden layer for better feature mapping
        # 128 * 4 * 4 = 2048 features
        self.fc = nn.Linear(128, n_classes)
        
        # Initialize weights
        self._init_weights()
        
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        
        x = self.gap(x)
        x = x.flatten(1)
        
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Using the same optimizer setup as the model to improve (which is fixed)
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
        # Start at lr, end at 0
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