import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return ['lr', 'momentum', 'batch', 'transform']

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 16))
        self.fc2 = nn.Linear(max(channels // reduction, 16), channels)

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y

class GhostBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dw_kernel_size=1, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.out_channels = out_channels
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // ratio, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels // ratio),
            nn.SiLU(inplace=True)
        )
        self.ghost_conv = nn.Conv2d(out_channels // ratio, out_channels, kernel_size=dw_kernel_size, padding=dw_kernel_size // 2, groups=out_channels // ratio, bias=False)

    def forward(self, x):
        x = self.cheap_conv(x)
        x = self.ghost_conv(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.transform = prm.get('transform', 'norm_32')
        
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        # Stage 1: 32x32 -> 16x16
        # 3 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )

        # Block 1: 48 -> 48 (16x16)
        self.block1 = nn.Sequential(
            GhostBlock(48, 48),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            SEBlock(48)
        )

        # Transition 1: 16x16 -> 8x8, 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )

        # Block 2: 96 -> 96 (8x8)
        self.block2 = nn.Sequential(
            GhostBlock(96, 96),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            SEBlock(96)
        )

        # Block 3: 96 -> 96 (8x8)
        self.block3 = nn.Sequential(
            GhostBlock(96, 96),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            SEBlock(96)
        )

        # Transition 2: 8x8 -> 4x4, 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )

        # Block 4: 192 -> 192 (4x4)
        self.block4 = nn.Sequential(
            GhostBlock(192, 192),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            SEBlock(192)
        )

        # Global Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, num_classes)

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
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.trans2(x)
        x = self.block4(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        # Use SGD with momentum, as it often generalizes better for 1-epoch training on CIFAR
        # High learning rate is crucial for 1-epoch training
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Calculate total steps for LR scheduling
        # Assuming standard CIFAR-10 dataset size (50,000)
        # We need to estimate batches. The DataLoader is passed to learn.
        # We will store the dataset length if possible, or estimate.
        # Since learn receives train_data (DataLoader), we can get len(train_data.dataset)
        # But we don't have access to it in train_setup.
        # We will define the scheduler in learn or assume a standard number of steps.
        # CIFAR-10: 50,000 images. Batch size is in prm.
        # Steps = 50000 / batch_size
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        # Cosine Annealing is often good, but for 1 epoch, a linear decay or constant might be safer.
        # Let's use Cosine Annealing to ensure a smooth finish.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=0
        )
        return self.optimizer

    def learn(self, train_data):
        self.train()
        # We need to update the scheduler.
        # The scheduler is created in train_setup with an estimated total steps.
        # If the actual number of batches differs, the schedule might be slightly off,
        # but it's a reasonable approximation for CIFAR-10.
        
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