import torch
import torch.nn as nn
import numpy as np


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        dropout_prob = prm['dropout']
        
        # [A15] Layout: 4 convolution stages reaching up to 256 channels.
        # Standard widths for such a CNN (often ResNet-like or Wide-ResNet-like but described as "4 stages"):
        # Stage 1: 64 channels
        # Stage 2: 128 channels
        # Stage 3: 256 channels
        # Stage 4: 256 channels (or sometimes 512, but "up to 256" implies max is 256)
        
        # The description mentions "residual-style skip connection in deeper feature stages".
        # A common interpretation of "4 convolution stages" with residuals is a ResNet-like structure.
        # Let's implement a ResNet-18 style or a simpler 4-stage CNN with residuals.
        # Given "up to 256 channels", a typical configuration is:
        # Conv1 -> 64
        # Stage 1 (Residuals) -> 64
        # Stage 2 (Residuals, Downsample) -> 128
        # Stage 3 (Residuals, Downsample) -> 256
        # Stage 4 (Residuals, Downsample) -> 256 (or 512? "up to 256" suggests 256 is the max)
        
        # Let's assume a structure similar to ResNet-18 but adapted to hit 256 max channels.
        # ResNet-18 standard: [64, 128, 256, 512]. 
        # If we must stay "up to 256", we might use [64, 128, 256, 256].
        # Or perhaps the "4 stages" are:
        # 1. Conv 3x3 -> 64
        # 2. Residual Block (64->64)
        # 3. Residual Block (64->128, downsample)
        # 4. Residual Block (128->256, downsample)
        # 5. Residual Block (256->256) ?? 
        # The prompt says "4 convolution stages". 
        # Let's interpret "4 convolution stages" as 4 groups of layers.
        # Group 1: 64 channels
        # Group 2: 128 channels
        # Group 3: 256 channels
        # Group 4: 256 channels
        
        # We will use BasicBlocks (2 conv layers) for each stage to satisfy "residual-style".
        # Depth: Standard ResNet-18 has 2 blocks per stage. Let's use 2 blocks per stage for a total of 8 residual blocks + initial conv.
        # This fits "deep feature stages".
        
        in_channels = in_shape[1]
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1: 64 channels
        self.stage1 = self._make_layer(64, 64, num_blocks=2, stride=1, dropout_prob=dropout_prob)
        
        # Stage 2: 128 channels, downsample
        self.stage2 = self._make_layer(64, 128, num_blocks=2, stride=2, dropout_prob=dropout_prob)
        
        # Stage 3: 256 channels, downsample
        self.stage3 = self._make_layer(128, 256, num_blocks=2, stride=2, dropout_prob=dropout_prob)
        
        # Stage 4: 256 channels, downsample? 
        # If we downsample here, spatial size becomes 4x4 (from 32x32).
        # If we don't downsample, it stays 8x8.
        # "Up to 256 channels" usually implies the final layer is 256.
        # Let's add a 4th stage with 256 channels. To maintain the "4 stages" count distinct from the initial conv, 
        # and considering standard ResNet downsampling happens at stages 2, 3, 4 (if 4 stages of blocks), 
        # let's assume Stage 4 also downsamples or is just another block group.
        # Given CIFAR-10 is 32x32:
        # Init: 32x32
        # Stage 1 (stride 1): 32x32
        # Stage 2 (stride 2): 16x16
        # Stage 3 (stride 2): 8x8
        # Stage 4 (stride 2): 4x4 -> This is very small. 
        # Alternatively, Stage 4 could be stride 1.
        # However, "4 convolution stages" often maps to the 4 groups in ResNet.
        # Let's use stride 2 for stage 4 to follow the standard ResNet pattern of downsampling in deeper stages, 
        # or keep it at 8x8 if we want to be safe? 
        # Actually, ResNet-18 on CIFAR-10 typically uses:
        # Conv 3x3 (32x32)
        # Layer1: 2 blocks, 64 ch, stride 1 (32x32)
        # Layer2: 2 blocks, 128 ch, stride 2 (16x16)
        # Layer3: 2 blocks, 256 ch, stride 2 (8x8)
        # Layer4: 2 blocks, 512 ch, stride 2 (4x4)
        # The prompt says "up to 256 channels". So we replace 512 with 256.
        # So Stage 4 will have 256 channels.
        
        self.stage4 = self._make_layer(256, 256, num_blocks=2, stride=2, dropout_prob=dropout_prob)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Compute linear layer size
        # Input to linear is 256 * 1 * 1 = 256
        self.fc = nn.Linear(256, out_shape[0])
        
        self._init_params()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_prob):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_channels, out_channels, stride, dropout_prob))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device).to(torch.float32), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Dropout as required by hyperparameters
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        # Residual connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        
        return out