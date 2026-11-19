import torch
import torch.nn as nn
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'transform'}

class InvertedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=False, expand=2):
        super(InvertedResBlock, self).__init__()
        self.use_residual = use_residual
        
        # Expand
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expand, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * expand),
            nn.SiLU(inplace=True)
        )
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels * expand, in_channels * expand, kernel_size=3, padding=1, groups=in_channels * expand, bias=False),
            nn.BatchNorm2d(in_channels * expand),
            nn.SiLU(inplace=True)
        )
        
        # Project
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * expand, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        
        # Determine spatial size
        # in_shape is usually (C, H, W) or (H, W, C). 
        # CIFAR is usually (3, 32, 32).
        # If in_shape[0] == 3, assume (C, H, W). Else assume (H, W, C).
        if in_shape[0] == 3:
            self.height, self.width = in_shape[1], in_shape[2]
        else:
            self.height, self.width = in_shape[0], in_shape[1]
            
        num_classes = out_shape[0] if out_shape else 10
        
        # Entry: 3 -> 64 (No pool, keep 32x32)
        # Using a slightly wider entry to capture features early
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 64 -> 64 (32x32)
        # 2 blocks
        self.stage1 = nn.Sequential(
            InvertedResBlock(64, 64, use_residual=False, expand=2),
            InvertedResBlock(64, 64, use_residual=True, expand=2)
        )
        
        # Transition: Pool + Channel Change
        # 64 -> 128 (16x16)
        self.trans1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        # Stage 2: 128 -> 128 (16x16)
        # 3 blocks
        self.stage2 = nn.Sequential(
            InvertedResBlock(128, 128, use_residual=False, expand=2),
            InvertedResBlock(128, 128, use_residual=True, expand=2),
            InvertedResBlock(128, 128, use_residual=True, expand=2)
        )
        
        # Transition: Pool + Channel Change
        # 128 -> 256 (8x8)
        self.trans2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # Stage 3: 256 -> 256 (8x8)
        # 2 blocks
        self.stage3 = nn.Sequential(
            InvertedResBlock(256, 256, use_residual=False, expand=2),
            InvertedResBlock(256, 256, use_residual=True, expand=2)
        )
        
        # Global Pool
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        # Keep it simple but effective.
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, num_classes)
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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        # Use SGD with momentum. 
        # We want a high initial LR that decays.
        # CIFAR-10 is small, so SGD often works better than Adam for single epoch if tuned right, 
        # but Adam is robust. Let's stick with SGD as it's common for these "one epoch" challenges 
        # to get high accuracy quickly if the LR is high enough.
        # Actually, let's use SGD with Nesterov momentum for stability.
        
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4,
            nesterov=True
        )
        
        # We need a learning rate schedule within the single epoch.
        # Cosine annealing is good.
        # We need to know the number of steps.
        # We can estimate it or use a LambdaLR.
        # Let's assume a standard CIFAR-10 dataset size (50,000) and batch size.
        # But we don't know the batch size in train_setup? 
        # The prompt says: "choose the hyperparameters... prm['batch'] is passed".
        # Wait, the prompt says: "Then give the hyperparameters in a <prm> block as JSON: {"lr": <float>, "batch": <16, 32, 64 or 128>, "momentum": <float in [0, 1)>, "transform": <"norm_32" or "norm_32_flip">}"
        # So `prm` will contain 'batch'.
        
        batch_size = prm['batch']
        # Estimate steps. CIFAR-10 has 50k images.
        # If the dataset provided is the full training set, steps = 50000 // batch_size.
        # To be safe, we can just use a simple linear decay or cosine based on a fixed number of steps if we can't know it exactly.
        # However, `learn` iterates over `train_data`. 
        # A common trick: use a LambdaLR that decays based on the current step relative to a max step.
        # Since we don't know the exact length of the DataLoader in train_setup, we can set a reasonable max_steps.
        # 50000 / 128 ~ 390 steps. 50000 / 16 ~ 3125 steps.
        # Let's use a LambdaLR that decays to 0.1 * lr by the end of the estimated epoch.
        
        # We can update the scheduler in learn if we want, but LambdaLR is stateless.
        # Let's define a lambda that takes step.
        # We'll assume max_steps is roughly 50000 // batch_size.
        max_steps = 50000 // batch_size
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=lambda step: max(0.1, 0.5 * (1 + math.cos(math.pi * step / max_steps)))
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()