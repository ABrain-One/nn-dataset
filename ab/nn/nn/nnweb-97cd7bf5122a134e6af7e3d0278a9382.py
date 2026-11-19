import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = self.act(y)
        return x * y

class GhostBlock(nn.Module):
    """
    Ghost module: 
    1. Standard Conv (1x1) to reduce channels.
    2. Depthwise Conv (3x3) to generate ghost features.
    3. SE attention applied to the output.
    4. Residual connection.
    """
    def __init__(self, in_ch, out_ch, stride=1, expansion=2):
        super().__init__()
        mid_ch = max(1, in_ch // expansion)
        
        # Primary generation
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        
        # Ghost generation (depthwise)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, stride=stride, groups=mid_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        # Attention
        self.se = SEBlock(out_ch)
        
        self.act = nn.SiLU(inplace=True)
        
        # Residual projection if needed
        self.residual = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.residual(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        
        x = self.se(x)
        
        x = x + identity
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.prm = prm
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3x3 Conv, BN, SiLU
        # Start with 48 channels to allow for deeper features without exploding params
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 -> 48 (2 blocks), AvgPool 2
        # Spatial: 32 -> 32 -> 16
        self.stage1 = nn.Sequential(
            GhostBlock(48, 48, stride=1, expansion=2),
            GhostBlock(48, 48, stride=1, expansion=2),
            nn.AvgPool2d(2)
        )
        
        # Stage 2: 48 -> 96 (2 blocks), AvgPool 2
        # Spatial: 16 -> 16 -> 8
        self.stage2 = nn.Sequential(
            GhostBlock(48, 96, stride=1, expansion=2),
            GhostBlock(96, 96, stride=1, expansion=2),
            nn.AvgPool2d(2)
        )
        
        # Stage 3: 96 -> 192 (2 blocks), AvgPool 2
        # Spatial: 8 -> 8 -> 4
        self.stage3 = nn.Sequential(
            GhostBlock(96, 192, stride=1, expansion=2),
            GhostBlock(192, 192, stride=1, expansion=2),
            nn.AvgPool2d(2)
        )
        
        # Stage 4: 192 -> 384 (1 block), AvgPool 2
        # Spatial: 4 -> 4 -> 2
        # Using 384 channels here to keep width high for the final GAP
        self.stage4 = nn.Sequential(
            GhostBlock(192, 384, stride=1, expansion=2),
            nn.AvgPool2d(2)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(384, n_classes)
        
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
        # Apply normalization if the harness doesn't do it (defensive)
        # Note: The prompt implies 'transform' is a hyperparam, usually handled by the data loader.
        # However, if 'transform' implies we *should* normalize in the model (like Design 2 did), 
        # we check it. But standard practice is that 'transform' is a config for the dataloader.
        # Model 1 and 3 didn't normalize in forward. Design 2 did.
        # Given CIFAR-10, raw pixels are [0,1]. Normalizing helps.
        # I will assume the harness applies the transform to the data. 
        # If I normalize here, I might double-normalize if the harness does it too.
        # Looking at Design 1: "No normalization in forward, trusting the harness".
        # Looking at Design 2: "if self.transform in ('norm_32'...): x = (x - 0.4914) / 0.2470".
        # To be safe and consistent with the "strongest" model (Model 1, which didn't normalize),
        # I will NOT normalize in forward. The 'transform' hyperparam is read but assumed to be for the dataloader.
        
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
        self.criterion = nn.CrossEntropyLoss()
        
        # SGD with momentum is generally more stable for 1-epoch training than Adam 
        # if the LR is tuned correctly. 
        # Model 1 used AdamW with lr=0.002. 
        # Model 3 used SGD with lr=0.01 (implied by default or prm).
        # Let's use SGD with a warmup/cosine schedule for better convergence in 1 epoch.
        
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        
        self.batch_size = prm['batch']
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        total_steps = len(train_data)
        
        # Warmup for first 5% of steps, then Cosine decay
        warmup_steps = max(1, int(total_steps * 0.05))
        
        # We will use a manual scheduler logic to ensure smooth decay
        # Cosine Annealing from base_lr to 0
        
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Learning Rate Schedule
            if i < warmup_steps:
                # Linear warmup
                lr_scale = (i + 1) / warmup_steps
            else:
                # Cosine decay
                progress = (i - warmup_steps) / max(1, total_steps - warmup_steps)
                lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            
            current_lr = self.prm['lr'] * lr_scale
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()