import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=4):
        super().__init__()
        hidden = max(1, out_ch // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, out_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is (B, C, H, W)
        se_out = self.se(x)
        # se_out is (B, C). Need to reshape to (B, C, 1, 1) for broadcasting.
        se_out = se_out.unsqueeze(2).unsqueeze(3)
        return x * se_out

class EfficientBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se_ratio=4):
        super().__init__()
        
        # Depthwise Conv
        self.dw_conv = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        
        # Pointwise Conv
        self.pw_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        
        # SE Block
        # Note: SE operates on the output of the pointwise conv (out_ch)
        self.se = SEBlock(in_ch, out_ch, reduction=se_ratio)
        
        self.act = nn.SiLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.skip(x)
        
        # Depthwise
        out = self.dw_conv(x)
        out = self.dw_bn(out)
        out = self.act(out)
        
        # Pointwise
        out = self.pw_conv(out)
        out = self.pw_bn(out)
        
        # SE
        out = self.se(out)
        
        out = out + identity
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.prm = prm
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        
        # Entry: Wide conv to capture features early
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 -> 64, size /2
        self.stage1 = nn.Sequential(
            EfficientBlock(32, 64, stride=1),
            EfficientBlock(64, 64, stride=2)
        )
        
        # Stage 2: 64 -> 128, size /2
        self.stage2 = nn.Sequential(
            EfficientBlock(64, 128, stride=1),
            EfficientBlock(128, 128, stride=2),
            EfficientBlock(128, 128, stride=1)
        )
        
        # Stage 3: 128 -> 256, size /2
        self.stage3 = nn.Sequential(
            EfficientBlock(128, 256, stride=1),
            EfficientBlock(256, 256, stride=2),
            EfficientBlock(256, 256, stride=1)
        )
        
        # Stage 4: 256 -> 512, size /2
        self.stage4 = nn.Sequential(
            EfficientBlock(256, 512, stride=1),
            EfficientBlock(512, 512, stride=2)
        )
        
        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == m.in_channels: # Depthwise
                    nn.init.kaiming_normal_(m.weight, mode='fan_in')
                else:
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
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.prm = prm
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Use SGD with momentum as per standard CIFAR practice for stability in 1 epoch
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        
        # Cosine annealing schedule
        # We assume a standard batch size to estimate steps
        batch_size = prm['batch']
        # CIFAR-10 has 50000 training samples
        num_steps = 50000 // batch_size
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_steps, eta_min=0.0)
        
        # Use the 'transform' hyperparameter to configure training behavior
        self.augment_flip = (prm['transform'] == 'norm_32_flip')
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        batch_size = self.prm['batch']
        num_steps = 50000 // batch_size
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Apply horizontal flip augmentation if specified in transform
            if self.augment_flip:
                # Flip 50% of the images in the batch
                flip_mask = torch.rand(inputs.size(0), device=inputs.device) > 0.5
                if flip_mask.any():
                    inputs[flip_mask] = torch.flip(inputs[flip_mask], dims=[3])
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            
            self.optimizer.step()
            self.scheduler.step()