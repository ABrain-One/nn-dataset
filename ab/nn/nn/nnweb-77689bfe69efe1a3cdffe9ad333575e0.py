import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(2, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(2, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class EfficientBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True, use_residual=True):
        super(EfficientBlock, self).__init__()
        self.use_residual = use_residual
        self.use_se = use_se
        
        # Depthwise Separable Conv
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True)
        
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
            
        # Projection for residual
        if use_residual:
            if stride != 1 or in_channels != out_channels:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.skip = nn.Identity()

    def forward(self, x):
        identity = x
        
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.pw_conv(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.act(out)
        
        if self.use_residual:
            identity = self.skip(identity)
            out = out + identity
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        
        self.transform_str = prm['transform']
        self.use_flip = 'flip' in self.transform_str
        
        # Initial Stem: Standard Conv to extract basic features
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 -> 64 (Stride 1)
        self.stage1 = nn.Sequential(
            EfficientBlock(32, 64, stride=1, use_se=False),
            EfficientBlock(64, 64, stride=1, use_se=False)
        )
        
        # Stage 2: 64 -> 128 (Stride 2)
        self.stage2 = nn.Sequential(
            EfficientBlock(64, 128, stride=2, use_se=True),
            EfficientBlock(128, 128, stride=1, use_se=True),
            EfficientBlock(128, 128, stride=1, use_se=True)
        )
        
        # Stage 3: 128 -> 256 (Stride 2)
        self.stage3 = nn.Sequential(
            EfficientBlock(128, 256, stride=2, use_se=True),
            EfficientBlock(256, 256, stride=1, use_se=True),
            EfficientBlock(256, 256, stride=1, use_se=True)
        )
        
        # Stage 4: 256 -> 512 (Stride 2)
        self.stage4 = nn.Sequential(
            EfficientBlock(256, 512, stride=2, use_se=True),
            EfficientBlock(512, 512, stride=1, use_se=True),
            EfficientBlock(512, 512, stride=1, use_se=True)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
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
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.training and self.use_flip:
            flip_idx = torch.rand(x.size(0), device=x.device) < 0.5
            if flip_idx.any():
                x[flip_idx] = torch.flip(x[flip_idx], dims=[3])
                
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
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = prm['batch']
        self.total_samples = 50000
        self.total_steps = self.total_samples // self.batch_size
        self.current_step = 0
        # Store the initial learning rate explicitly to avoid relying on 'initial_lr' key
        self.initial_lr = prm['lr']

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            if self.total_steps > 0:
                t = self.current_step / self.total_steps
                # Cosine decay using stored initial_lr
                lr = self.initial_lr * 0.5 * (1 + math.cos(math.pi * t))
                self.optimizer.param_groups[0]['lr'] = lr
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.current_step += 1