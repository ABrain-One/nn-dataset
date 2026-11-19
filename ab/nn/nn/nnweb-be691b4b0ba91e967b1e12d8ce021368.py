import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        reduction_dim = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, reduction_dim)
        self.fc2 = nn.Linear(reduction_dim, channels)

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1)
        y = y.view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = self.fc2(y)
        y = F.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y

class MobileInvertedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_residual=True):
        super(MobileInvertedBlock, self).__init__()
        self.use_residual = use_residual and (in_channels == out_channels and stride == 1)
        
        # 1x1 expand
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 3x3 depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 1x1 project
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.se = SEBlock(out_channels, reduction=8)

    def forward(self, x):
        input_x = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + input_x
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        
        # Derive spatial dimensions
        self.h, self.w = in_shape[2], in_shape[3]
        
        # Channel progression: Avoid 64-128-256-512. Use 48-96-192-384
        # This keeps parameter count low while maintaining capacity.
        
        # Stem: Large kernel to capture context early, no pooling
        # 3 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 -> 48 (Spatial: H/2)
        # One block with stride 2
        self.stage1 = nn.Sequential(
            MobileInvertedBlock(48, 48, stride=2, use_residual=False)
        )
        
        # Stage 2: 48 -> 96 (Spatial: H/2)
        # Two blocks, first with stride 1, second with stride 1 (No spatial reduction here to keep features rich)
        self.stage2 = nn.Sequential(
            MobileInvertedBlock(48, 96, stride=1, use_residual=False),
            MobileInvertedBlock(96, 96, stride=1, use_residual=True)
        )
        
        # Stage 3: 96 -> 192 (Spatial: H/4)
        self.stage3 = nn.Sequential(
            MobileInvertedBlock(96, 192, stride=2, use_residual=False),
            MobileInvertedBlock(192, 192, stride=1, use_residual=True)
        )
        
        # Stage 4: 192 -> 384 (Spatial: H/4)
        self.stage4 = nn.Sequential(
            MobileInvertedBlock(192, 384, stride=1, use_residual=False),
            MobileInvertedBlock(384, 384, stride=1, use_residual=True)
        )
        
        # Head
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier: Simple linear layer
        self.classifier = nn.Sequential(
            nn.Linear(384, out_shape[0])
        )
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(device)

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
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        # Use SGD with momentum as it's often more stable for one-epoch training on CIFAR
        # Learning rate schedule: Cosine decay within the single epoch
        # We need to estimate total steps.
        # CIFAR-10 has 50000 images.
        # Batch size is provided in prm.
        batch_size = prm.get('batch', 128)
        
        # Use the 'transform' hyperparameter to determine dataset size
        # 'norm_32' implies standard CIFAR-10 (50k train)
        # 'norm_32_flip' implies augmented CIFAR-10 (100k train)
        transform = prm.get('transform', 'norm_32')
        if transform == 'norm_32_flip':
            total_images = 100000
        else:
            total_images = 50000
            
        total_steps = total_images // batch_size
        
        self.total_steps = total_steps
        self.base_lr = prm['lr']
        self.momentum = prm['momentum']
        
        # Initialize optimizer
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.base_lr, 
            momentum=self.momentum, 
            weight_decay=1e-4
        )
        
        # Cosine LR scheduler
        # We will implement a simple cosine decay manually in the loop or use LambdaLR
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_steps, eta_min=1e-5)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Move to device
        self.to(self.device)
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        # train_data is a DataLoader
        # We assume it iterates over the entire dataset once.
        # We need to track steps to update the scheduler correctly.
        # Since CosineAnnealingLR steps per epoch by default, and we have 1 epoch,
        # we need to step it for every batch to get the decay curve across the epoch.
        
        # Note: The standard CosineAnnealingLR expects `step()` to be called once per epoch.
        # To get per-batch decay, we should use LambdaLR or step it every batch if T_max is total_steps.
        # Let's re-create the scheduler to be per-batch.
        # Actually, CosineAnnealingLR with T_max=total_steps and calling step() every batch works perfectly.
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            
            self.optimizer.step()
            
            # Step the scheduler
            self.scheduler.step()