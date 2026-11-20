import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_depthwise=False):
        super().__init__()
        self.stride = stride
        self.use_depthwise = use_depthwise
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        if use_depthwise:
            self.conv1 = SepConv(in_channels, in_channels, 3, 1, 1)
            self.conv2 = SepConv(in_channels, out_channels, 3, stride, 1)
        else:
            self.conv1 = ConvBNAct(in_channels, in_channels, 3, 1, 1)
            self.conv2 = ConvBNAct(in_channels, out_channels, 3, stride, 1)
            
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = nn.SiLU(inplace=True)(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        
        # Entry: 3x3 conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 64 -> 128
        self.stage1 = nn.Sequential(
            BottleneckBlock(64, 64, 1, use_depthwise=False),
            BottleneckBlock(64, 64, 1, use_depthwise=False),
            BottleneckBlock(64, 128, 2, use_depthwise=False)
        )
        
        # Stage 2: 128 -> 256
        self.stage2 = nn.Sequential(
            BottleneckBlock(128, 128, 1, use_depthwise=True),
            BottleneckBlock(128, 128, 1, use_depthwise=True),
            BottleneckBlock(128, 256, 2, use_depthwise=True)
        )
        
        # Stage 3: 256 -> 512
        self.stage3 = nn.Sequential(
            BottleneckBlock(256, 256, 1, use_depthwise=True),
            BottleneckBlock(256, 256, 1, use_depthwise=True),
            BottleneckBlock(256, 512, 2, use_depthwise=True)
        )
        
        # Stage 4: 512 -> 512 (No downsample, keep resolution for features)
        self.stage4 = nn.Sequential(
            BottleneckBlock(512, 512, 1, use_depthwise=True),
            BottleneckBlock(512, 512, 1, use_depthwise=True),
            BottleneckBlock(512, 512, 1, use_depthwise=True)
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(512, num_classes)
        )
        
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
        # If the transform indicates flipping is allowed, we can apply random horizontal flip
        # This acts as an in-model data augmentation step.
        if self.transform == 'norm_32_flip':
            if self.training:
                # Random horizontal flip
                x = torch.flip(x, dims=[3])
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with momentum and weight decay for stability
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        
        # Simple cosine decay within the single epoch
        total_steps = 50000 // prm['batch']
        self.total_steps = total_steps
        self.step = 0
        
        self.base_lr = prm['lr']
        self.min_lr = prm['lr'] * 0.01

    def _adjust_lr(self):
        # Cosine schedule
        progress = self.step / self.total_steps
        lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr.item()

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
            self.step += 1
            self._adjust_lr()