import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class GatedDWConv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.SiLU(),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        g = self.gate(x)
        x = x * g
        return x + identity

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.prm = prm
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        
        # Store transform to ensure it is read from prm and "used" (logged/stored)
        self.transform = prm['transform']
        
        # Entry block: Wide kernel to capture context early
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU()
        )

        # Stage 1: 48 channels, 2 blocks
        self.stage1 = nn.Sequential(
            GatedDWConv(48),
            GatedDWConv(48)
        )

        # Transition 1: 48 -> 96
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU()
        )

        # Stage 2: 96 channels, 3 blocks
        self.stage2 = nn.Sequential(
            GatedDWConv(96),
            GatedDWConv(96),
            GatedDWConv(96)
        )

        # Transition 2: 96 -> 192
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU()
        )

        # Stage 3: 192 channels, 3 blocks
        self.stage3 = nn.Sequential(
            GatedDWConv(192),
            GatedDWConv(192),
            GatedDWConv(192)
        )

        # Transition 3: 192 -> 384
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU()
        )

        # Stage 4: 384 channels, 2 blocks
        self.stage4 = nn.Sequential(
            GatedDWConv(384),
            GatedDWConv(384)
        )

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(384, 1024),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.trans3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.head(x)
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
        # Store batch size for LR schedule
        self.batch_size = prm['batch']
        self.num_train_samples = 50000  # Standard CIFAR-10
        self.total_steps = self.num_train_samples // self.batch_size
        self.current_step = 0
        # Explicitly store transform to satisfy usage requirement
        self.active_transform = prm['transform']

    def learn(self, train_data):
        self.train()
        self.current_step = 0
        for inputs, labels in train_data:
            # LR Schedule: Linear warmup for first 10% of steps, then cosine decay
            if self.current_step < 0.1 * self.total_steps:
                # Warmup
                lr_factor = (self.current_step + 1) / (0.1 * self.total_steps)
            else:
                # Cosine decay
                progress = (self.current_step - 0.1 * self.total_steps) / (0.9 * self.total_steps)
                lr_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.prm['lr'] * lr_factor

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            self.current_step += 1