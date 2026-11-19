import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.dropout_prob = prm['dropout']
        
        in_channels = in_shape[1]
        H, W = in_shape[2], in_shape[3]
        
        # ResNet-style architecture with BasicBlock
        # Designed to be deep enough for good representation but efficient for 1-epoch training
        # Channels: 64 -> 128 -> 256 -> 512
        # Spatial sizes: H/2 -> H/4 -> H/8 -> H/16
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 64 channels, stride 1 (input size HxW)
        self.stage1 = nn.Sequential(
            BasicBlock(64, 64, stride=1, dropout_prob=self.dropout_prob),
            BasicBlock(64, 64, stride=1, dropout_prob=self.dropout_prob)
        )
        
        # Stage 2: 128 channels, stride 2 (size H/2 x W/2)
        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, dropout_prob=self.dropout_prob),
            BasicBlock(128, 128, stride=1, dropout_prob=self.dropout_prob)
        )
        
        # Stage 3: 256 channels, stride 2 (size H/4 x W/4)
        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, dropout_prob=self.dropout_prob),
            BasicBlock(256, 256, stride=1, dropout_prob=self.dropout_prob)
        )
        
        # Stage 4: 512 channels, stride 2 (size H/8 x W/8)
        self.stage4 = nn.Sequential(
            BasicBlock(256, 512, stride=2, dropout_prob=self.dropout_prob),
            BasicBlock(512, 512, stride=1, dropout_prob=self.dropout_prob)
        )
        
        # Global Average Pooling and Classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=self.dropout_prob),
            nn.Linear(512, out_shape[0])
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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

    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out