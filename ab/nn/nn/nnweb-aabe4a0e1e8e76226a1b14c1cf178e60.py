import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SqueezeExcite(nn.Module):
    """
    Lightweight SE block for channel attention.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MBConvBlock(nn.Module):
    """
    MobileNetV2-style block: Expand -> Depthwise -> Project + Residual + SE
    """
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4, use_se=True):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden_dim = int(round(in_ch * expand_ratio))
        
        # Expand
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        # Project
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        
        # SE Block
        self.se = SqueezeExcite(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        input = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        x = self.se(x)
        
        if self.use_residual:
            x = x + input
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10
        
        # Stem: 3 -> 48 (No pool, keep spatial dims high for a bit)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 48 channels, 32x32 -> 32x32
        # 2 blocks
        self.stage1 = nn.Sequential(
            MBConvBlock(48, 48, stride=1, expand_ratio=4, use_se=False),
            MBConvBlock(48, 48, stride=1, expand_ratio=4, use_se=False)
        )
        
        # Transition 1: 48 -> 96, 32x32 -> 16x16
        self.trans1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 96 channels, 16x16 -> 16x16
        # 3 blocks
        self.stage2 = nn.Sequential(
            MBConvBlock(96, 96, stride=1, expand_ratio=4, use_se=True),
            MBConvBlock(96, 96, stride=1, expand_ratio=4, use_se=True),
            MBConvBlock(96, 96, stride=1, expand_ratio=4, use_se=True)
        )
        
        # Transition 2: 96 -> 192, 16x16 -> 8x8
        self.trans2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 192 channels, 8x8 -> 8x8
        # 4 blocks
        self.stage3 = nn.Sequential(
            MBConvBlock(192, 192, stride=1, expand_ratio=4, use_se=True),
            MBConvBlock(192, 192, stride=1, expand_ratio=4, use_se=True),
            MBConvBlock(192, 192, stride=1, expand_ratio=4, use_se=True),
            MBConvBlock(192, 192, stride=1, expand_ratio=4, use_se=True)
        )
        
        # Transition 3: 192 -> 384, 8x8 -> 4x4
        self.trans3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 384 channels, 4x4 -> 4x4
        # 2 blocks
        self.stage4 = nn.Sequential(
            MBConvBlock(384, 384, stride=1, expand_ratio=4, use_se=True),
            MBConvBlock(384, 384, stride=1, expand_ratio=4, use_se=True)
        )
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
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
        # Apply normalization if the transform suggests it
        # The harness usually applies transforms, but if we are given raw data and 'norm_32' is selected, 
        # we might need to handle it. However, typically the harness applies the transform.
        # If the harness does NOT apply the transform, we should. 
        # Given the ambiguity, and that Model 1/2/3 didn't explicitly normalize in forward 
        # (except Design 2 did), I will assume the harness applies the transform.
        # However, to be safe and match the "norm_32" logic if the harness is passive:
        # Actually, looking at Design 2, it explicitly normalized. Design 1 did not.
        # The prompt says "transform": <"norm_32" or "norm_32_flip">. 
        # Usually, this implies the DataLoader applies it. 
        # I will NOT normalize in forward to avoid double normalization if the harness does it.
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.trans3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # SGD with momentum is generally more stable for 1-epoch training than AdamW 
        # if tuned correctly, but AdamW was the winner in Model 1.
        # Model 1: AdamW, lr=0.002, momentum=0.75 (beta1).
        # Let's stick with AdamW as it seems to have worked best in the dataset (74.66%).
        # We will use a cosine schedule.
        
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Estimate total steps
        # CIFAR-10 has 50,000 training images.
        batch_size = prm.get('batch', 128)
        total_steps = 50000 // batch_size
        
        # Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=prm['lr'] * 0.01
        )
        
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()