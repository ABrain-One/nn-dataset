import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as T

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio=4, use_se=True):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        # Ensure hidden_dim is at least 1 to avoid zero channels in conv
        if hidden_dim < 1:
            hidden_dim = 1
            
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        if use_se:
            self.se = SEBlock(hidden_dim)
        else:
            self.se = nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            x = x + self.project(self.se(self.conv(x)))
        else:
            x = self.project(self.se(self.conv(x)))
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.prm = prm
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        
        # Entry: No pool, stride 1 to keep spatial resolution high initially
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Block configuration: (out_channels, num_blocks, stride)
        # Avoiding 64-128-256-512 progression
        # Using a progression: 32 -> 64 -> 128 -> 256 -> 256
        self.stage1 = self._make_stage(32, 64, 2, 1, expand_ratio=4, use_se=False)
        self.stage2 = self._make_stage(64, 128, 2, 2, expand_ratio=4, use_se=True)
        self.stage3 = self._make_stage(128, 256, 4, 2, expand_ratio=4, use_se=True)
        self.stage4 = self._make_stage(256, 256, 2, 1, expand_ratio=4, use_se=True)
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(256, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
        self._setup_transform(prm)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, expand_ratio, use_se):
        layers = []
        # First block handles stride
        layers.append(InvertedResidual(in_channels, out_channels, stride, expand_ratio, use_se))
        for _ in range(1, num_blocks):
            layers.append(InvertedResidual(out_channels, out_channels, 1, expand_ratio, use_se))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
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

    def _setup_transform(self, prm):
        transform_type = prm.get('transform', 'norm_32')
        
        # Define normalization stats
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        
        if transform_type == 'norm_32':
            self.transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.ToTensor(),
                T.Normalize(mean, std)
            ])
        elif transform_type == 'norm_32_flip':
            self.transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std)
            ])
        else:
            # Default fallback
            self.transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std)
            ])

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.prm = prm
        self._setup_transform(prm)
        
        lr = prm['lr']
        momentum = prm['momentum']
        
        # Use SGD with momentum
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=lr, 
            momentum=momentum,
            weight_decay=5e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        batch_size = self.prm['batch']
        
        # Estimate total steps
        try:
            total_samples = len(train_data)
        except:
            total_samples = 50000 

        total_steps = max(1, total_samples // batch_size)
        
        # Warmup for 5% of steps
        warmup_steps = max(1, int(total_steps * 0.05))
        
        for i, (inputs, labels) in enumerate(train_data):
            # Handle inputs
            if isinstance(inputs, torch.Tensor):
                # If it's a tensor, check if it's already normalized.
                # Heuristic: If mean is close to 0.5, it's likely raw [0,1].
                # If mean is close to 0, it's likely normalized.
                # However, applying transform to a tensor batch is complex.
                # We assume if it's a tensor, it's already processed by the DataLoader.
                # But to strictly "use" the transform hyperparameter in a meaningful way
                # if the data is raw, we'd need to detect it.
                # For robustness in this benchmark context, we assume DataLoader provides
                # transformed data if 'transform' is specified in the framework setup.
                # If not, and we get raw tensors, we normalize them manually.
                
                # Check if tensor is in [0,1] range (raw) or normalized.
                # If min >= 0 and max <= 1, it's likely raw.
                if inputs.min() >= 0 and inputs.max() <= 1:
                    # Normalize manually using stats from self.transform
                    # Extract stats
                    # Since we defined them in _setup_transform, let's store them.
                    # For now, assume standard CIFAR stats.
                    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(inputs.device)
                    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1).to(inputs.device)
                    inputs = (inputs - mean) / std
                # If it's already normalized (has negative values or >1), leave it.
                
                inputs = inputs.to(self.device)
            else:
                # PIL or other
                inputs = self.transform(inputs).to(self.device)
            
            labels = labels.to(self.device)
            
            # Learning rate schedule
            if i < warmup_steps:
                lr_scale = (i + 1) / warmup_steps
            else:
                # Ensure denominator is not zero
                remaining_steps = total_steps - warmup_steps
                if remaining_steps <= 0:
                    progress = 1.0
                else:
                    progress = (i - warmup_steps) / remaining_steps
                # Cosine decay
                lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.prm['lr'] * lr_scale

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()