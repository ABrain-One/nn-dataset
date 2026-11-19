import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.silu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = F.silu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        self.batch_size = prm.get('batch', 128)
        
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        
        # Stem: Standard 3x3 Conv to establish spatial features
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: 64 channels, 32x32
        self.stage1 = nn.Sequential(
            Block(64, 64, stride=1),
            Block(64, 64, stride=1)
        )
        
        # Stage 2: 128 channels, 16x16
        self.stage2 = nn.Sequential(
            Block(64, 128, stride=2),
            Block(128, 128, stride=1)
        )
        
        # Stage 3: 256 channels, 8x8
        self.stage3 = nn.Sequential(
            Block(128, 256, stride=2),
            Block(256, 256, stride=1)
        )
        
        # Stage 4: 384 channels, 4x4
        # Reduced from 512 to 384 to save parameters.
        # Use Depthwise-Separable here to increase capacity/depth without exploding params
        self.stage4 = nn.Sequential(
            SepConv(256, 384),
            nn.SiLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(384, num_classes)

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

    def forward(self, x):
        if self.transform == 'norm_32':
            x = (x - 0.5) / 0.5
        elif self.transform == 'norm_32_flip':
            x = (x - 0.5) / 0.5
            x = torch.flip(x, dims=[-1])
            
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
        # Cosine schedule within the epoch
        # Estimate total steps based on CIFAR-10 dataset size (50,000) and batch size
        total_steps = 50000 // self.batch_size
        def lr_lambda(step):
            return 0.5 * (1 + math.cos(math.pi * step / total_steps))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()