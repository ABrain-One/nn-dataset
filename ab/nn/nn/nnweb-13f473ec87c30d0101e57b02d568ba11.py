import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class BasicBlock(nn.Module):
    """
    Basic residual block for CIFARResidualNet.
    Structure: Conv3x3 -> BN -> ReLU -> Dropout -> Conv3x3 -> BN
    Shortcut: Identity (or 1x1 Conv if shapes mismatch)
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Optional Dropout as per description
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device

        channel_number = in_shape[1]
        image_size = in_shape[2]
        class_number = out_shape[0]

        # Architecture: CIFARResidualNet
        # Stem: 3x3 Conv2d, norm, activation
        # Stage 1: Residual blocks at 32x32 (width 64)
        # Stage 2: Residual blocks at 16x16 (width 128)
        # Stage 3: Residual blocks at 8x8 (width 256)
        
        # Standard widths for this type of small residual net on CIFAR
        # Using 64, 128, 256 as standard progression
        width = 64
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(channel_number, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 32x32
        self.stage1 = self._make_layer(width, width, num_blocks=2, stride=1, dropout_p=prm['dropout'])
        
        # Stage 2: 16x16
        self.stage2 = self._make_layer(width, width * 2, num_blocks=2, stride=2, dropout_p=prm['dropout'])
        
        # Stage 3: 8x8
        self.stage3 = self._make_layer(width * 2, width * 4, num_blocks=2, stride=2, dropout_p=prm['dropout'])
        
        # Head: GAP and Classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(width * 4, class_number)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_p):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, strides[0], dropout_p))
        in_channels = out_channels
        for stride in strides[1:]:
            layers.append(BasicBlock(in_channels, out_channels, stride, dropout_p))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)

        lr = prm['lr']
        momentum = prm['momentum']

        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()