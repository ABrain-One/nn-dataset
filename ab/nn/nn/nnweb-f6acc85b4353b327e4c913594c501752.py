import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')
        
        # Input normalization based on transform
        if self.transform == 'norm_32':
            self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
        else: # norm_32_flip
            self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)

        in_channels = 3
        # Reduced initial channels to save parameters
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet-like stages with reduced channel counts
        # Stage 1: 32 -> 64
        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        # Stage 2: 64 -> 128
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        # Stage 3: 128 -> 256
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        # Stage 4: 256 -> 512 (Keeping 512 for final feature richness, but previous stages are lighter)
        # Let's check params:
        # Conv1: 3*32*3*3 = 864
        # L1 (2 blocks, 32->64):
        #   Block1: Conv(32,64,3x3)=18432, Conv(64,64,3x3)=36864, Downsample(32,64,1x1)=2048
        #   Block2: Conv(64,64,3x3)=36864, Conv(64,64,3x3)=36864
        #   Total L1 ~ 131k
        # L2 (2 blocks, 64->128):
        #   Block1: Conv(64,128,3x3)=73728, Conv(128,128,3x3)=147456, Downsample(64,128,1x1)=8192
        #   Block2: Conv(128,128,3x3)=147456, Conv(128,128,3x3)=147456
        #   Total L2 ~ 524k
        # L3 (2 blocks, 128->256):
        #   Block1: Conv(128,256,3x3)=294912, Conv(256,256,3x3)=589824, Downsample(128,256,1x1)=32768
        #   Block2: Conv(256,256,3x3)=589824, Conv(256,256,3x3)=589824
        #   Total L3 ~ 2.1M
        # L4 (2 blocks, 256->512):
        #   Block1: Conv(256,512,3x3)=1179648, Conv(512,512,3x3)=2359296, Downsample(256,512,1x1)=131072
        #   Block2: Conv(512,512,3x3)=2359296, Conv(512,512,3x3)=2359296
        #   Total L4 ~ 8.4M
        # This is too big. I need to reduce L4 or L3.
        
        # Let's try a smaller ResNet18 variant:
        # L3: 128 -> 128 (keep channels same? No, standard is double)
        # Let's try:
        # L1: 32->64 (2 blocks)
        # L2: 64->128 (2 blocks)
        # L3: 128->128 (2 blocks) -- No, let's do 128->256 but maybe fewer blocks or smaller kernel?
        # Standard ResNet18 on CIFAR often uses 64, 128, 256.
        # Params for 64, 128, 256 (2 blocks each):
        # L1 (64): ~131k (calculated above with 32 start)
        # L2 (128): ~524k
        # L3 (256): ~2.1M
        # Total so far: ~2.75M.
        # If I add L4 (512), it adds ~8.4M. Total ~11M.
        
        # To stay under 5M, I should drop L4 or reduce L3.
        # Let's try:
        # L1: 32->64 (2 blocks)
        # L2: 64->128 (2 blocks)
        # L3: 128->256 (2 blocks)
        # FC: 256 -> out
        # Total ~2.75M + FC. This is safe.
        
        # However, deeper networks often perform better. Can I make L4 smaller?
        # If I do L3: 128->256, and L4: 256->256 (no channel increase)?
        # L4 (256, 2 blocks):
        #   Block1: Conv(256,256,3x3)=589824, Conv(256,256,3x3)=589824, Downsample(256,256,1x1)=65536 (stride 2)
        #   Block2: Conv(256,256,3x3)=589824, Conv(256,256,3x3)=589824
        #   Total L4 ~ 2.4M.
        # Total: 2.75M + 2.4M = 5.15M. Still slightly over.
        
        # Let's reduce L3 to 1 block? Or reduce channels in L3?
        # Let's try:
        # L1: 32->64 (2 blocks)
        # L2: 64->128 (2 blocks)
        # L3: 128->128 (2 blocks) -- Wait, usually we double.
        # Let's try:
        # L1: 32->64 (2 blocks)
        # L2: 64->128 (2 blocks)
        # L3: 128->256 (2 blocks)
        # No L4.
        # This is effectively ResNet-14 (32, 64, 128, 256).
        # Params: ~2.75M.
        # This is well under 5M.
        
        # To get closer to 5M and potentially better accuracy, I can increase blocks in L3 or L2.
        # Let's do:
        # L1: 32->64 (2 blocks)
        # L2: 64->128 (2 blocks)
        # L3: 128->256 (4 blocks)
        # L3 with 4 blocks:
        #   Block1: ~0.9M (downsample)
        #   Block2-4: 3 * (589824*2) = 3.5M
        #   Total L3 ~ 4.4M.
        # Total: 131k + 524k + 4.4M = 5.05M. Slightly over.
        
        # Let's do:
        # L1: 32->64 (2 blocks)
        # L2: 64->128 (2 blocks)
        # L3: 128->256 (3 blocks)
        # L3 with 3 blocks:
        #   Block1: ~0.9M
        #   Block2-3: 2 * 1.18M = 2.36M
        #   Total L3 ~ 3.26M.
        # Total: 131k + 524k + 3.26M = 3.9M.
        # This is safe.
        
        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(256, out_shape[0])
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        lr = prm['lr']
        momentum = prm['momentum']
        
        # Use SGD with momentum, standard for CIFAR-10
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=5e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Cosine annealing schedule for the single epoch
        # CIFAR-10 has 50,000 images.
        batch_size = prm['batch']
        self.num_batches = 50000 // batch_size
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_batches, eta_min=lr*0.01)
        
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
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