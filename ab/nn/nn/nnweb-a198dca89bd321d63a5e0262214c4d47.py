import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum', 'transform'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.prm = prm
        in_channels = in_shape[1]
        out_classes = out_shape[0]

        # Entry: 3x3 conv, BN, SiLU
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 2x Inverted Residual blocks, 48->96 channels, stride 2
        self.stage1 = nn.Sequential(
            InvertedResidual(48, 96, 2, 4),
            InvertedResidual(96, 96, 1, 4)
        )

        # Stage 2: 3x Inverted Residual blocks, 96->192 channels, stride 2
        self.stage2 = nn.Sequential(
            InvertedResidual(96, 192, 2, 4),
            InvertedResidual(192, 192, 1, 4),
            InvertedResidual(192, 192, 1, 4)
        )

        # Stage 3: 2x Inverted Residual blocks, 192->384 channels, stride 1
        self.stage3 = nn.Sequential(
            InvertedResidual(192, 384, 1, 4),
            InvertedResidual(384, 384, 1, 4)
        )

        # SE Block
        self.se = SEBlock(384, reduction=8)

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Linear(384, out_classes)

        # Weight initialization
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
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.se(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        # Use SGD with momentum (momentum in [0, 1))
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.transform = prm['transform']
        self.base_lr = prm['lr']
        return self.optimizer

    def learn(self, train_data):
        self.train()
        # One pass over train_data
        # Use cosine-like decay within the epoch
        total_batches = len(train_data)
        
        for i, (inputs, labels) in enumerate(train_data):
            # Cosine decay of learning rate within the epoch
            t = i / max(1, total_batches - 1)
            lr_scale = 0.5 * (1 + torch.cos(torch.tensor(t * 3.141592653589793)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * lr_scale

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            self.optimizer.step()


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(in_channels * expansion))
        
        self.use_shortcut = (stride == 1) and (in_channels == out_channels)
        
        layers = []
        
        # Expand
        layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.SiLU(inplace=True))
        
        # Depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.SiLU(inplace=True))
        
        # Project
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y