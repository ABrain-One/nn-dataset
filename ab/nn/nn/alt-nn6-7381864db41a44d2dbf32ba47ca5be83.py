import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GELU

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class ICInitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ICInitBlock, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class PSPBlock(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PSPBlock, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1, bias=False)
            )
            for size in pool_sizes
        ])
        self.bottleneck = nn.Conv2d(in_channels + in_channels // len(pool_sizes) * len(pool_sizes), in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        pooled_features = [F.interpolate(stage(x), size=size, mode='bilinear', align_corners=False) for stage in self.stages]
        return F.relu(self.bottleneck(torch.cat([x] + pooled_features, dim=1)))

class ICNetClassification(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(ICNetClassification, self).__init__()
        self.init_block = ICInitBlock(in_channels, 64)
        self.psp_block = PSPBlock(64, pool_sizes=[1, 2, 4, 5])
        self.head_block = nn.Conv2d(64, 128, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.init_block(x)
        x = self.psp_block(x)
        x = self.head_block(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.model = ICNetClassification(num_classes=out_shape[0], in_channels=in_shape[1])
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.dropout = prm['dropout']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=0.0001                                         
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

addon_accuracy: 0.9709