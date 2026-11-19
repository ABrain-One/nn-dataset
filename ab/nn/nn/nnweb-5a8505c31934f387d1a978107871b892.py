import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        # Shortcut connection: must match spatial dimensions and channels of the output of the main path
        # If stride != 1 or channels change, we need a projection.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply shortcut to identity
        identity = self.shortcut(identity)
        
        out += identity
        out = self.act(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.prm = prm
        
        # in_shape is (C, H, W) or (B, C, H, W). We assume standard CIFAR-10 input (3, 32, 32)
        # but we derive H, W to be robust.
        if len(in_shape) == 3:
            c_in, h, w = in_shape
        else:
            _, c_in, h, w = in_shape
            
        self.h = h
        self.w = w

        # Start with a larger receptive field stem to capture global structure early
        # Reduced channels to fit parameter budget
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )

        # Residual Blocks
        # Stage 1: 32x32 -> 32x32
        self.layer1 = nn.Sequential(
            ResidualBlock(48, 48, stride=1),
            ResidualBlock(48, 48, stride=1)
        )

        # Stage 2: 32x32 -> 16x16
        self.layer2 = nn.Sequential(
            ResidualBlock(48, 96, stride=2),
            ResidualBlock(96, 96, stride=1)
        )

        # Stage 3: 16x16 -> 8x8
        self.layer3 = nn.Sequential(
            ResidualBlock(96, 192, stride=2),
            ResidualBlock(192, 192, stride=1)
        )

        # Stage 4: 8x8 -> 8x8 (Deeper features)
        self.layer4 = nn.Sequential(
            ResidualBlock(192, 192, stride=1),
            ResidualBlock(192, 192, stride=1)
        )

        # Global Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        # Reduced hidden size to fit parameter budget
        self.classifier = nn.Sequential(
            nn.Linear(192, 384),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(384, out_shape[0])
        )

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        # Use SGD with momentum
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Store transform info
        self.transform = prm.get('transform', 'norm_32_flip')
        
        # Estimate total steps for scheduling
        # CIFAR-10 has 50,000 training samples.
        n_samples = 50000
        batch_size = prm.get('batch', 128)
        self.total_steps = max(1, n_samples // batch_size)
        self.current_step = 0
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        # Linear Warmup for first 5% of steps, then Cosine Decay
        warmup_steps = int(0.05 * self.total_steps)
        
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Update LR
            if self.current_step < warmup_steps:
                lr_scale = (self.current_step + 1) / warmup_steps
                new_lr = self.prm['lr'] * lr_scale
            else:
                progress = (self.current_step - warmup_steps) / max(1, self.total_steps - warmup_steps)
                progress = min(1.0, max(0.0, progress))
                lr_scale = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
                new_lr = self.prm['lr'] * lr_scale
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.current_step += 1

        # Reset step for potential re-calls
        self.current_step = 0