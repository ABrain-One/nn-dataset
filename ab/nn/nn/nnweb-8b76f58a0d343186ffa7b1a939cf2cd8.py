import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'transform'}

class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted bottleneck with SE-like channel gating."""
    def __init__(self, in_ch, out_ch, stride, expand_ratio, se_ratio=4):
        super(InvertedResidual, self).__init__()
        expanded = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch
        
        self.conv1 = nn.Conv2d(in_ch, expanded, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded)
        
        self.conv2 = nn.Conv2d(expanded, expanded, 3, stride, 1, groups=expanded, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded, max(1, expanded // se_ratio), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, expanded // se_ratio), expanded, 1),
            nn.Sigmoid()
        )
        
        self.conv3 = nn.Conv2d(expanded, out_ch, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.silu(x, inplace=True)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.silu(x, inplace=True)
        
        x = x * self.se(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.use_residual:
            x = x + input
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32')
        
        # Input is (B, C, H, W)
        in_ch = in_shape[1]
        self.in_h = in_shape[2]
        self.in_w = in_shape[3]
        self.n_classes = out_shape[0]
        
        # 1. Stem: Wide kernel for better feature extraction at start
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # 2. Stages: Non-standard channel progression (32 -> 64 -> 128 -> 256)
        # Using Inverted Residuals with expansion ratio 4
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=4),
            InvertedResidual(64, 64, stride=1, expand_ratio=4)
        )
        
        self.stage2 = nn.Sequential(
            InvertedResidual(64, 128, stride=2, expand_ratio=4),
            InvertedResidual(128, 128, stride=1, expand_ratio=4),
            InvertedResidual(128, 128, stride=1, expand_ratio=4)
        )
        
        self.stage3 = nn.Sequential(
            InvertedResidual(128, 256, stride=2, expand_ratio=4),
            InvertedResidual(256, 256, stride=1, expand_ratio=4),
            InvertedResidual(256, 256, stride=1, expand_ratio=4)
        )
        
        # 3. Head: Global pooling + Classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, self.n_classes)
        )
        
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
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use SGD with Nesterov momentum for stability
        # Learning rate will be scheduled in learn
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'], 
            momentum=prm['momentum'], 
            nesterov=True,
            weight_decay=1e-4
        )
        return self.optimizer

    def learn(self, train_data):
        self.train()
        
        # Calculate total steps for cosine schedule
        # Note: train_data is a DataLoader. We need to know the length.
        # If length is not available (e.g. iterable dataset), we assume a standard count or handle it dynamically.
        # For CIFAR-10 standard DataLoader, len() works.
        try:
            total_steps = len(train_data)
        except TypeError:
            # Fallback if len() is not supported
            total_steps = 1000 

        # Cosine Annealing within the single epoch
        # We will update the learning rate manually or use a scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=0.0)
        
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
            scheduler.step()