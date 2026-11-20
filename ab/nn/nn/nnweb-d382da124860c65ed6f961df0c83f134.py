import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, drop_path_prob=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.act = nn.SiLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride, stride=stride) if stride != 1 else nn.Identity(),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )

        self.drop_path = nn.Identity() if drop_path_prob == 0.0 else DropPath(drop_path_prob)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.drop_path(self.bn2(self.conv2(x)))
        x = x + shortcut
        x = self.act(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        num_classes = out_shape[0] if out_shape else 10
        in_channels = in_shape[1] if len(in_shape) > 1 else 3

        # Store hyperparameters for use in learn/train_setup
        self.batch_size = prm['batch']
        self.lr = prm['lr']
        self.momentum = prm['momentum']
        self.transform = prm.get('transform', 'norm_32_flip')

        # Reduced channel widths to stay under 5M parameters
        # Stem: 3 -> 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 64 -> 64 (2 blocks)
        self.stage1 = nn.Sequential(
            BasicBlock(64, 64, stride=1, drop_path_prob=0.0),
            BasicBlock(64, 64, stride=1, drop_path_prob=0.0)
        )

        # Stage 2: 64 -> 128 (2 blocks)
        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, drop_path_prob=0.1),
            BasicBlock(128, 128, stride=1, drop_path_prob=0.1)
        )

        # Stage 3: 128 -> 256 (2 blocks)
        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, drop_path_prob=0.2),
            BasicBlock(256, 256, stride=1, drop_path_prob=0.2)
        )

        # Final Projection: 256 -> 384
        self.final_proj = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(384, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_proj(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        
        # Use SGD with momentum
        # Use batch size to scale weight decay
        wd = 1e-4 * (self.batch_size / 256.0)
        
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=wd
        )
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()