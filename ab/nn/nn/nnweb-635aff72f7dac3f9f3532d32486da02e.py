import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1, act=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SE(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.SiLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class EfficientBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se_ratio=4):
        super().__init__()
        self.stride = stride
        self.bn = nn.BatchNorm2d(in_ch)
        
        if in_ch == out_ch and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        # Depthwise
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True)
        )
        
        # Pointwise
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        
        self.se = SE(out_ch, se_ratio)

    def forward(self, x):
        x = self.bn(x)
        shortcut = self.shortcut(x)
        
        x = self.dw(x)
        x = self.pw(x)
        x = self.se(x)
        
        return x + shortcut

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 64, kernel_size=3, stride=1, padding=1, act=nn.SiLU),
            ConvBNAct(64, 64, kernel_size=3, stride=1, padding=1, act=nn.SiLU),
        )
        
        # Stages
        # Stage 1: 32x32 -> 32x32
        self.stage1 = nn.Sequential(
            EfficientBlock(64, 64, stride=1),
            EfficientBlock(64, 64, stride=1),
            EfficientBlock(64, 64, stride=1),
        )
        
        # Stage 2: 32x32 -> 16x16
        self.stage2 = nn.Sequential(
            EfficientBlock(64, 128, stride=2),
            EfficientBlock(128, 128, stride=1),
            EfficientBlock(128, 128, stride=1),
            EfficientBlock(128, 128, stride=1),
        )
        
        # Stage 3: 16x16 -> 8x8
        self.stage3 = nn.Sequential(
            EfficientBlock(128, 256, stride=2),
            EfficientBlock(256, 256, stride=1),
            EfficientBlock(256, 256, stride=1),
            EfficientBlock(256, 256, stride=1),
        )
        
        # Head
        self.head_bn = nn.BatchNorm2d(256)
        self.head_act = nn.SiLU(inplace=True)
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.head_bn(x)
        x = self.head_act(x)
        x = self.head_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.prm = prm
        
        # Optimizer
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Schedule setup
        self.total_steps = None # Set in learn
        self.current_step = 0
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        # Estimate total steps for cosine schedule
        # train_data is a DataLoader
        if hasattr(train_data, 'batch_size'):
            batch_size = train_data.batch_size
        else:
            batch_size = self.prm.get('batch', 128)
            
        # Assume standard CIFAR-10 size (50k) for schedule estimation, 
        # but we iterate over whatever is provided.
        # To be robust, we'll just count as we go and apply a cosine decay based on an estimated total.
        # If the dataset size is not known, we assume 50000 samples.
        # A more robust way: if train_data has __len__, use it.
        try:
            total_samples = len(train_data.dataset)
        except:
            total_samples = 50000
            
        total_steps = max(1, total_samples // batch_size)
        
        base_lr = self.prm['lr']
        
        # Use transform hyperparameter to select augmentation strategy
        # If 'norm_32_flip' is selected, we apply random horizontal flips to the input.
        # This utilizes the parameter and provides data augmentation which helps generalization.
        use_flip = (self.prm.get('transform') == 'norm_32_flip')
        
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Apply augmentation based on hyperparameter
            if use_flip:
                # Random horizontal flip
                if torch.rand(1).item() > 0.5:
                    inputs = torch.flip(inputs, dims=[3])
            
            # Cosine LR Schedule
            # t goes from 0 to 1
            t = self.current_step / total_steps
            # Cosine decay to 0 (or near 0)
            # Standard cosine: lr = base_lr * 0.5 * (1 + cos(pi * t))
            # We might want to keep a small floor, but 0 is fine for 1 epoch.
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * t))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.current_step += 1