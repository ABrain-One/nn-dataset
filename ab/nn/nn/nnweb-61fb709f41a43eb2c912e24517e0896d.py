import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.dropout_prob = prm['dropout']
        self.momentum = prm['momentum']
        self.lr = prm['lr']

        in_channels = in_shape[1]
        
        # Stem: 3x3 Conv2d, norm, activation
        # Standard choice: 64 output channels for CIFAR-10 ResNet variants
        stem_out_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_out_channels),
            nn.ReLU(inplace=True)
        )

        # Stage 1: residual blocks at 32x32 (no downsampling)
        # Standard depth for small ResNet: 2 blocks per stage
        self.stage1 = nn.Sequential(
            ResidualBlock(stem_out_channels, stem_out_channels, stride=1, dropout_prob=self.dropout_prob),
            ResidualBlock(stem_out_channels, stem_out_channels, stride=1, dropout_prob=self.dropout_prob)
        )

        # Stage 2: residual blocks at 16x16 (downsampling)
        # Width doubles: 64 -> 128
        stage2_out_channels = 128
        self.stage2 = nn.Sequential(
            ResidualBlock(stem_out_channels, stage2_out_channels, stride=2, dropout_prob=self.dropout_prob),
            ResidualBlock(stage2_out_channels, stage2_out_channels, stride=1, dropout_prob=self.dropout_prob)
        )

        # Stage 3: residual blocks at 8x8 (downsampling)
        # Width doubles: 128 -> 256
        stage3_out_channels = 256
        self.stage3 = nn.Sequential(
            ResidualBlock(stage2_out_channels, stage3_out_channels, stride=2, dropout_prob=self.dropout_prob),
            ResidualBlock(stage3_out_channels, stage3_out_channels, stride=1, dropout_prob=self.dropout_prob)
        )

        # Global Average Pooling and Classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(stage3_out_channels, out_shape[0])

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
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout_prob):
        super().__init__()
        self.stride = stride
        self.needs_projection = (in_channels != out_channels) or (stride != 1)

        # Block: 3x3 Conv -> Norm -> Act -> Dropout -> 3x3 Conv -> Norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Optional Dropout2d
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        # Shortcut projection if needed
        if self.needs_projection:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(identity)

        out += identity
        out = F.relu(out, inplace=True)
        return out