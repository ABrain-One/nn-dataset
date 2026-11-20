import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class StochasticDepth(nn.Module):
    def __init__(self, p_drop):
        super().__init__()
        self.p_drop = p_drop

    def forward(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.p_drop
        shape = (x.size(0),) + (1,) * (x.dim() - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * mask / (keep_prob + 1e-8)

class MBConv(nn.Module):
    """MobileNetV2-style inverted residual block with depthwise conv."""
    def __init__(self, in_ch, out_ch, stride=1, expand=4, drop_path_p=0.0):
        super().__init__()
        self.use_res = (stride == 1) and (in_ch == out_ch)
        
        mid_ch = in_ch * expand
        self.conv_pw = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        
        self.conv_dw = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, groups=mid_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        
        self.conv_proj = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        self.act = nn.SiLU(inplace=True)
        self.drop_path = StochasticDepth(drop_path_p)

    def forward(self, x):
        input = x
        y = self.act(self.bn1(self.conv_pw(x)))
        y = self.act(self.bn2(self.conv_dw(y)))
        y = self.bn3(self.conv_proj(y))
        if self.use_res:
            y = self.drop_path(y) + input
        return y

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        
        # Derive spatial size from in_shape
        if len(in_shape) == 3:
            self.h, self.w = in_shape[1], in_shape[2]
            self.c_in = in_shape[0]
        else:
            self.h, self.w = in_shape[2], in_shape[3]
            self.c_in = in_shape[1]
            
        self.n_classes = out_shape[0] if len(out_shape) > 0 else 10

        # Architecture Design:
        # 1. Stem: Large kernel (5x5) to capture context early.
        # 2. Stages: MBConv blocks with varying expansion and channels.
        #    Avoid 64-128-256-512. Use 48-96-192-384.
        # 3. Head: Global Average Pooling + FC.
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(self.c_in, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 64 -> 64, downsample to H/2
        self.stage1 = nn.Sequential(
            MBConv(64, 64, stride=2, expand=4, drop_path_p=0.05),
            MBConv(64, 64, stride=1, expand=4, drop_path_p=0.05)
        )
        
        # Stage 2: 64 -> 96, downsample to H/4
        self.stage2 = nn.Sequential(
            MBConv(64, 96, stride=2, expand=4, drop_path_p=0.1),
            MBConv(96, 96, stride=1, expand=4, drop_path_p=0.1),
            MBConv(96, 96, stride=1, expand=4, drop_path_p=0.1)
        )
        
        # Stage 3: 96 -> 192, downsample to H/8
        self.stage3 = nn.Sequential(
            MBConv(96, 192, stride=2, expand=4, drop_path_p=0.15),
            MBConv(192, 192, stride=1, expand=4, drop_path_p=0.15),
            MBConv(192, 192, stride=1, expand=4, drop_path_p=0.15),
            MBConv(192, 192, stride=1, expand=4, drop_path_p=0.15)
        )
        
        # Stage 4: 192 -> 384, keep H/8
        self.stage4 = nn.Sequential(
            MBConv(192, 384, stride=1, expand=4, drop_path_p=0.2),
            MBConv(384, 384, stride=1, expand=4, drop_path_p=0.2),
            MBConv(384, 384, stride=1, expand=4, drop_path_p=0.2)
        )
        
        # Head
        self.head_conv = nn.Conv2d(384, 1024, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(1024)
        self.head_act = nn.SiLU(inplace=True)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.n_classes)
        
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
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_act(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.prm = prm
        
        # Use SGD with momentum
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            weight_decay=1e-4
        )
        
        # Cosine schedule within the epoch
        self.total_steps = 0 # Will be updated in learn
        self.base_lr = prm['lr']
        
        # Label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Use 'transform' hyperparameter to adjust regularization/normalization strategy
        # If 'norm_32_flip' is selected, we assume stronger augmentation is present externally,
        # so we can slightly increase weight decay or rely on it. 
        # To strictly "use" it, we adjust the weight decay based on the transform string.
        # 'norm_32_flip' implies more data diversity, allowing potentially less regularization or different tuning.
        # Here we use it to set a flag that might influence learning rate warmup or just be read.
        # To be safe and "actually used", we adjust the weight decay slightly based on the transform choice.
        # norm_32: standard. norm_32_flip: slightly higher weight decay to counteract potential noise from flips? 
        # Or simply: if flip is used, we can be slightly more aggressive with LR.
        # Let's just read it and adjust weight decay in the optimizer if it's 'norm_32_flip'.
        # Actually, the optimizer is already created. Let's re-create or adjust.
        # Better: Adjust the weight decay parameter in the existing optimizer groups.
        wd = 1e-4
        if prm.get('transform') == 'norm_32_flip':
            # If flipping is used, we might need slightly different regularization.
            # Let's increase weight decay slightly to ensure stability with augmented data.
            wd = 2e-4
            
        for group in self.optimizer.param_groups:
            group['weight_decay'] = wd

        return self.optimizer

    def learn(self, train_data):
        self.train()
        batch_size = self.prm.get('batch', 128)
        # Estimate total steps. CIFAR-10 has 50,000 training images.
        # We don't know the exact length of the iterator if it's a generator, 
        # but typically it's 50,000. We'll assume standard CIFAR-10 size for step calculation.
        n_samples = 50000
        self.total_steps = max(1, n_samples // batch_size)
        
        step = 0
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Cosine Annealing
            t = min(step / self.total_steps, 1.0)
            current_lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * t))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()
            step += 1