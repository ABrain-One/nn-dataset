import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class ParallelSeparableBlock(nn.Module):
    """
    A block consisting of N parallel branches.
    Each branch: DW Conv 3x3 -> BN -> SiLU -> 1x1 Conv -> BN
    Output: Average of all branches.
    """
    def __init__(self, channels, n_branches=2):
        super().__init__()
        self.branches = nn.ModuleList()
        for _ in range(n_branches):
            branch = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels)
            )
            self.branches.append(branch)
        self.n_branches = n_branches

    def forward(self, x):
        outs = []
        for branch in self.branches:
            outs.append(branch(x))
        # Average the parallel branches
        return torch.stack(outs, dim=0).mean(dim=0)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: Large kernel to capture context early (inspired by Model 2)
        # Use 5x5 to get spatial reduction or just large context.
        # Let's keep spatial size same for now, just wide context.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 96 channels
        self.stage1 = nn.Sequential(
            ParallelSeparableBlock(96, n_branches=2),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Transition 96 -> 192
        self.trans1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 192 channels
        self.stage2 = nn.Sequential(
            ParallelSeparableBlock(192, n_branches=3),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Transition 192 -> 384
        self.trans2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 384 channels
        self.stage3 = nn.Sequential(
            ParallelSeparableBlock(384, n_branches=2),
            # No pool here to keep resolution for GAP
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, n_classes)
        )
        
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
        # Apply normalization if needed
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW as in Model 1
        # Map momentum to beta1
        beta1 = prm['momentum']
        # Clamp beta1 to valid range for Adam (0, 0.999)
        beta1 = max(0.0, min(0.999, beta1))
        
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(beta1, 0.999),
            weight_decay=1e-4
        )
        
        self.batch_size = prm.get('batch', 128)
        return self.optimizer
    
    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
        
        # Cosine schedule
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
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            
            self.optimizer.step()
            scheduler.step()