import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock (2x 3x3 Conv).
    For CIFAR, the first conv has stride 1. Downsampling is handled by the second conv
    if downsample is not None, or by setting stride=2 in the first conv if downsample is None (standard CIFAR ResNet).
    However, the prompt asks for "same kind of blocks... as described".
    Description: "Blocks use 3x3 Conv2d, norm, activation, optional Dropout2d..."
    Standard CIFAR ResNet uses stride 1 in the first conv and stride 2 in the second if downsampling, 
    OR stride 2 in the first conv. 
    Let's stick to the most common CIFAR ResNet implementation (He et al. 2015):
    If downsample:
      Conv3x3 (stride=1) -> BN -> ReLU
      Conv3x3 (stride=2) -> BN
    Else:
      Conv3x3 (stride=1) -> BN -> ReLU
      Conv3x3 (stride=1) -> BN
    Shortcut: Identity or 1x1 Conv (stride=2) + BN.
    
    Wait, the prompt says "Stem: 3x3 Conv2d, norm, activation. Stage 1: residual blocks at 32x32. Stage 2: residual blocks at 16x16. Stage 3: residual blocks at 8x8."
    This implies downsampling happens between stages.
    Standard CIFAR ResNet:
    Stem: 3x3 Conv, BN, ReLU (no pooling)
    Stage 1: 32x32
    Stage 2: 16x16 (downsample in first block of stage)
    Stage 3: 8x8 (downsample in first block of stage)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Optional Dropout2d as per description
        if dropout_p > 0:
            self.drop = nn.Dropout2d(dropout_p)
        else:
            self.drop = nn.Identity()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # Standard choice: 1x1 conv for shortcut
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.drop(out) # Apply dropout after activation

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class CIFARResidualNet(nn.Module):
    def __init__(self, num_classes, in_channels, depth=20, width=16, dropout_p=0.0):
        super(CIFARResidualNet, self).__init__()
        
        # Determine number of blocks per stage
        # depth is total number of layers (including convs).
        # For BasicBlock, each block has 2 convs.
        # Total convs = depth.
        # Stem = 1 conv.
        # Head = 1 linear (not counted in depth usually, but let's assume depth refers to conv layers in the network body).
        # Standard CIFAR ResNet depth is 20, 32, 56, etc.
        # If depth=20: (20-2)/6 = 3 blocks per stage.
        # If depth=32: (32-2)/6 = 5 blocks per stage.
        # Let's assume the 'depth' hyperparameter isn't in supported_hyperparameters, so we must pick a standard one.
        # The prompt says "Where the description leaves a detail open, choose the standard choice for that architecture."
        # The description mentions "CIFARResidualNet". A very common small one is ResNet-20.
        # Let's hardcode depth=20 and width=16 as standard for this specific "small" description if not provided.
        # However, we must use `prm` for supported params. `depth` and `width` are NOT in supported_hyperparameters.
        # So I will use standard values: depth=20, width=16.
        
        if (depth - 2) % 6 != 0:
            raise ValueError("Depth should be 6n+2 for BasicBlock")
        
        num_blocks = (depth - 2) // 6
        self.in_channels = width

        # Stem: 3x3 Conv, BN, ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 32x32
        self.stage1 = self._make_layer(width, num_blocks, stride=1)
        
        # Stage 2: 16x16
        self.stage2 = self._make_layer(width, num_blocks, stride=2)
        
        # Stage 3: 8x8
        self.stage3 = self._make_layer(width, num_blocks, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(width, num_classes)

        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride, dropout_p=0.0))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):
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

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        channel_number = in_shape[1]
        class_number = out_shape[0]
        
        # We must use 'lr' and 'momentum' from prm.
        # We do not have 'depth' or 'width' in supported_hyperparameters, so we use standard defaults.
        # The description says "small residual network". ResNet-20 is standard.
        self.model = CIFARResidualNet(num_classes=class_number, in_channels=channel_number, depth=20, width=16, dropout_p=0.0)
        
        # Read hyperparameters to satisfy the rule "use only names the file already reads from prm"
        # and "each of those names ... must still be read from prm and actually used".
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

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