import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    """
    MobileNetV2-style block: Expand -> Depthwise -> Project
    """
    def __init__(self, in_channels, out_channels, stride, use_res=True):
        super(InvertedResidual, self).__init__()
        self.use_res = use_res and (stride == 1) and (in_channels == out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.use_res = False

        # Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * 4),
            nn.SiLU(inplace=True)
        )
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=3, stride=stride, padding=1, groups=in_channels * 4, bias=False),
            nn.BatchNorm2d(in_channels * 4),
            nn.SiLU(inplace=True)
        )
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        input = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_res:
            x = x + input
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        n_classes = out_shape[0] if out_shape else 10
        
        # Input shape is (C, H, W)
        # We derive spatial dimensions from in_shape
        # CIFAR-10 is 32x32, but we must not hardcode.
        # in_shape[1] is Channels, in_shape[2] is H, in_shape[3] is W
        # Wait, standard PyTorch input is (N, C, H, W). 
        # The prompt says "in_shape". Usually this is (C, H, W) or (H, W, C).
        # Looking at Model 1: `c_in = 3`. It uses `nn.Conv2d(c_in, ...)`.
        # Looking at Model 2: `in_channels = in_shape[1]`. This implies in_shape is (N, C, H, W) or (C, H, W) where C is at index 1?
        # Let's check Model 1 again. It hardcodes `c_in = 3`.
        # Let's check Model 2: `in_channels = in_shape[1]`. If in_shape is (3, 32, 32), index 1 is 32. That would be wrong for channels.
        # If in_shape is (32, 32, 3) (H, W, C), index 1 is 32.
        # If in_shape is (N, C, H, W), index 1 is C.
        # Let's look at the "MODEL TO IMPROVE". It uses `c_in = 3`. It ignores in_shape for channels.
        # However, Model 2 uses `in_channels = in_shape[1]`.
        # Let's assume in_shape is (C, H, W) or (H, W, C)?
        # Actually, standard PyTorch datasets provide (N, C, H, W).
        # If `in_shape` is passed as a tuple, it's often (C, H, W) or (H, W, C).
        # Let's look at Model 4: `in_channels = in_shape[1]`.
        # If in_shape is (3, 32, 32), in_shape[1] is 32. That would mean 32 input channels? Unlikely for CIFAR.
        # If in_shape is (32, 32, 3), in_shape[1] is 32.
        # If in_shape is (N, C, H, W), in_shape[1] is C.
        # Given the ambiguity, and that Model 1 hardcodes 3, I will assume `in_shape` might be (C, H, W) or just use 3 for CIFAR.
        # BUT, the prompt says "Never hard-code the input resolution".
        # It does NOT say "Never hard-code input channels".
        # However, to be safe, I will try to infer channels.
        # If in_shape is (C, H, W), C is in_shape[0].
        # If in_shape is (H, W, C), C is in_shape[2].
        # If in_shape is (N, C, H, W), C is in_shape[1].
        # Let's look at the "MODEL TO IMPROVE" again. It ignores in_shape for channels.
        # I will use `in_shape[1]` as channels, assuming the format is (N, C, H, W) or similar where C is index 1, 
        # OR I will just use 3. 
        # Actually, looking at Model 2: `in_channels = in_shape[1]`. 
        # If I use `in_shape[1]` and it turns out to be H, the network will fail or be wrong.
        # Let's look at the provided code for Model 1. It uses `c_in = 3`.
        # I will use `in_shape[1]` as the number of channels, assuming the convention used in Model 2 and 4.
        # If in_shape is (3, 32, 32), this fails. 
        # Let's check the context of "in_shape". In many of these challenges, `in_shape` is (C, H, W).
        # If in_shape is (3, 32, 32), in_shape[1] is 32.
        # If Model 2 uses `in_shape[1]` and works (69% acc), then in_shape[1] MUST be the channel count.
        # Therefore, in_shape is likely (N, C, H, W) or (C, H, W) where C is at index 1? No, (C,H,W) has C at 0.
        # So in_shape is likely (N, C, H, W).
        
        in_channels = in_shape[1]
        
        # Architecture:
        # Stem: 3x3 Conv, BN, SiLU
        # Stages of InvertedResiduals (MobileNetV2 style)
        # Channels: 32 -> 64 -> 128 -> 256
        # This avoids the 64-128-256-512 progression (stops at 256)
        # Uses Depthwise Convs (from Model 1)
        # Uses SiLU (from Model 1)
        # Uses BatchNorm (from Model 1)
        # Uses Residual connections (standard)
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 32 -> 32 (1 block)
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 32, stride=1)
        )
        
        # Stage 2: 32 -> 64 (2 blocks, first with stride 2)
        self.stage2 = nn.Sequential(
            InvertedResidual(32, 64, stride=2),
            InvertedResidual(64, 64, stride=1)
        )
        
        # Stage 3: 64 -> 128 (3 blocks, first with stride 2)
        self.stage3 = nn.Sequential(
            InvertedResidual(64, 128, stride=2),
            InvertedResidual(128, 128, stride=1),
            InvertedResidual(128, 128, stride=1)
        )
        
        # Stage 4: 128 -> 256 (4 blocks, first with stride 2)
        self.stage4 = nn.Sequential(
            InvertedResidual(128, 256, stride=2),
            InvertedResidual(256, 256, stride=1),
            InvertedResidual(256, 256, stride=1),
            InvertedResidual(256, 256, stride=1)
        )
        
        # Head
        # 1x1 Conv to expand channels for classification?
        # MobileNetV2 uses a 1x1 conv to 1280. That's too many params.
        # Let's use a 1x1 conv to n_classes directly after Global Avg Pool.
        # Or a small FC.
        # 256 channels * 4x4 (if 32x32 input, 4 downsamples -> 2x2? 
        # 32 -> 32 (stride 1) -> 16 (stride 2) -> 8 (stride 2) -> 4 (stride 2).
        # So spatial size is 4x4.
        # 256 * 4 * 4 = 4096.
        # Linear(4096, 10) is small.
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, n_classes)
        
        self._initialize_weights()
        
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
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW with momentum mapped to beta1
        # Model 1 (74.66%) used AdamW with lr=0.002, momentum=0.75
        # We use SGD with momentum because it is often more stable for 1-epoch training
        # when combined with a proper LR schedule.
        # Let's stick to the strong signal from Model 1: AdamW.
        # But Model 1's success might be due to the specific architecture + AdamW.
        # Let's try SGD with a cosine schedule, which is the gold standard for 1-epoch CIFAR.
        # Actually, let's look at the hyperparameters provided.
        # If I use SGD, I should use a high LR.
        # If I use AdamW, I should use a low LR.
        
        # Let's use SGD with momentum as it's robust.
        # But wait, Model 1 used AdamW.
        # Let's try a hybrid: AdamW with a cosine schedule.
        
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4
        )
        
        # Store batch size
        self.batch_size = prm.get('batch', 128)
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000
        
        # Cosine schedule within 1 epoch
        # Start at lr, end at 0
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            
            self.optimizer.step()
            scheduler.step()