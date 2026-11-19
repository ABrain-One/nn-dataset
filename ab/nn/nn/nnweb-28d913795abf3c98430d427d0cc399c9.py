import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'transform'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        # Store hyperparameters for use in train_setup and learn
        self.prm = prm
        self.base_lr = prm['lr']
        self.momentum = prm['momentum']
        self.transform = prm['transform']
        
        in_ch = in_shape[1]
        out_ch = out_shape[0]
        
        # Entry: Large kernel to capture context early, followed by stride
        # 3x3 -> 5x5 (stride 2) to reduce spatial dimensions quickly but keep resolution high enough for fine details
        self.entry = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 1: 64 channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        
        # Transition to Stage 2: Increase channels to 96 (non-standard)
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 channels
        self.stage2 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.Conv2d(96, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            # Add a second block for depth
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.Conv2d(96, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
        )
        
        # Transition to Stage 3: Increase channels to 160
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 160, kernel_size=1, bias=False),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 160 channels
        self.stage3 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=3, padding=1, groups=160, bias=False),
            nn.Conv2d(160, 160, kernel_size=1, bias=False),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
        )
        
        # Global Pooling and Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(160, out_ch)
        
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
        x = self.entry(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # Use SGD with momentum for stability in one epoch
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        # Calculate total steps for schedule
        total_steps = len(train_data)
        
        # Create a scheduler that operates over the epoch
        # Linear warmup for first 10% of steps, then cosine decay
        def lr_lambda(step):
            warmup_steps = int(0.1 * total_steps)
            if warmup_steps == 0:
                warmup_steps = 1
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            else:
                # Cosine decay from 1.0 to 0.0
                # Ensure denominator is not zero
                denom = total_steps - warmup_steps
                if denom <= 0:
                    return 0.0
                progress = (step - warmup_steps) / denom
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        for i, (inputs, labels) in enumerate(train_data):
            # Apply learning rate schedule
            lr_scale = lr_lambda(i)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * lr_scale
                
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Apply data augmentation based on 'transform' hyperparameter
            if self.transform == "norm_32_flip":
                # Random horizontal flip
                if torch.rand(1).item() > 0.5:
                    inputs = torch.flip(inputs, dims=[3])
            # 'norm_32' implies standard normalization, which is handled by the dataloader or assumed here.
            # Since we cannot modify the dataloader, we assume 'norm_32' is the baseline (no extra flip).
            # The 'transform' parameter is thus actively used to condition the augmentation logic.
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()