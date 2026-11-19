import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from typing import List, Tuple

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class BasicBlock(nn.Module):
    """
    ResNet Basic Block:
    BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        
        out = self.downsample(out)
        out += identity
        out = self.relu(out)
        return out

class FractalBlock(nn.Module):
    """
    From Model 1: A block that branches into two sub-blocks and averages them.
    Uses depthwise-separable convs (DwConv -> PwConv) for efficiency and strong feature mixing.
    """
    def __init__(self, n_columns: int, channels: int):
        super().__init__()
        self.n_columns = int(n_columns)
        channels = int(channels)
        
        # Depthwise Separable Conv Block: DwConv -> BN -> Act -> PwConv -> BN -> Act
        self.conv_layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

        if self.n_columns > 1:
            self.left = FractalBlock(self.n_columns - 1, channels)
            self.right_1 = FractalBlock(self.n_columns - 1, channels)
            self.right_2 = FractalBlock(self.n_columns - 1, channels)

    def forward(self, x):
        if self.n_columns == 1:
            return self.conv_layer(x)
        
        out_left = self.left(x)
        out_right = self.right_2(self.right_1(x))
        
        # Simple average aggregation (similar to FractalDropPath with p=0)
        return (out_left + out_right) / 2.0

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        
        # Determine image size from in_shape (assuming in_shape is [C, H, W])
        # CIFAR-10 is usually [3, 32, 32]
        self.image_size = in_shape[2] if len(in_shape) > 2 else 32

        # Determine transformation
        transform_name = prm.get('transform', 'norm_32_flip')
        
        # CIFAR-10 normalization stats
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]

        if transform_name == 'norm_32_flip':
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.Normalize(mean, std)
            ])
        else: # norm_32
            self.transform = T.Compose([
                T.Normalize(mean, std)
            ])

        # Entry: Large kernel (5x5) to capture wide context early (inspired by Model 2's 5x5 start)
        # But we keep stride 1 to preserve resolution, then downsample later.
        # We use 64 channels to keep parameter count low.
        self.entry = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: ResNet-like blocks (Standard depth)
        # 2 blocks, 64 channels
        self.stage1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )
        
        # Downsample
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Transition to 128
        self.proj1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: Fractal Blocks (Wide parallel processing)
        # We use Fractal blocks here to increase effective depth without linearly increasing sequential depth
        # which can hinder gradient flow in 1 epoch.
        # 2 Fractal blocks, 128 channels
        self.stage2 = nn.Sequential(
            FractalBlock(3, 128),
            FractalBlock(3, 128)
        )
        
        # Downsample
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Transition to 256
        self.proj2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: ResNet blocks for stability and capacity
        self.stage3 = nn.Sequential(
            BasicBlock(256, 256, stride=1),
            BasicBlock(256, 256, stride=1)
        )
        
        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        # Add a small FC layer for better generalization?
        # Model 1 used direct FC. Model 2 used large FCs.
        # Let's use a single FC layer with Dropout.
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
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
        # Apply transformation (normalization and optional flip)
        x = self.transform(x)
        
        x = self.entry(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.proj1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.proj2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        lr = prm['lr']
        momentum = prm['momentum']
        
        # Use SGD with momentum.
        # Weight decay is important for regularization.
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=1e-4
        )
        
        # We need a learning rate schedule for 1 epoch.
        # Cosine annealing is good.
        # But we need to know the number of steps.
        # We will set the scheduler in learn, or here if we know batch size.
        # prm['batch'] is available.
        # CIFAR-10 has 50,000 training images.
        # steps = 50000 / batch_size
        batch_size = prm.get('batch', 128)
        steps = 50000 // batch_size
        
        # Cosine annealing from lr to 0
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=steps, eta_min=0)
        
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