import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        dropout_prob = prm['dropout']
        
        # CIFAR-10 has 10 classes, but we use out_shape[0]
        num_classes = out_shape[0]
        in_channels = in_shape[1]
        
        # Standard ResNet-20 style widths for CIFAR
        # Stage 1: 32x32, 64 channels
        # Stage 2: 16x16, 128 channels
        # Stage 3: 8x8, 256 channels
        widths = [64, 128, 256]
        
        # Stem: 3x3 Conv, Norm, Activation
        # Standard choice: No padding in stem for CIFAR (32->32), or padding=1? 
        # CIFAR input is 32x32. 
        # If Stem is 3x3, padding=1 keeps 32x32.
        # If Stem is 3x3, padding=0 reduces to 30x30. 
        # The description says "Stage 1: residual blocks at 32x32". 
        # So Stem must preserve 32x32. Thus padding=1.
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: Residual blocks at 32x32
        # Standard ResNet-20 has 3 blocks per stage. 
        # Let's use 3 blocks per stage for a "small residual network" typical of CIFAR.
        # Block structure: 3x3 Conv -> Norm -> Act -> (Optional Drop) -> 3x3 Conv -> Norm -> (Optional Drop)
        # Residual connection: Identity if same shape, else 1x1 Conv + Norm
        
        self.stage1 = self._make_stage(widths[0], widths[0], num_blocks=3, dropout_prob=dropout_prob)
        
        # Stage 2: Residual blocks at 16x16
        # Downsampling happens here. Standard is stride=2 in the first conv of the block.
        self.stage2 = self._make_stage(widths[0], widths[1], num_blocks=3, dropout_prob=dropout_prob, downsample=True)
        
        # Stage 3: Residual blocks at 8x8
        # Downsampling happens here.
        self.stage3 = self._make_stage(widths[1], widths[2], num_blocks=3, dropout_prob=dropout_prob, downsample=True)
        
        # Ends with GAP and classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(widths[2], num_classes)
        
        self._init_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, dropout_prob, downsample=False):
        layers = []
        
        # First block might need downsampling
        if downsample:
            # Standard choice: 1x1 conv with stride 2 for shortcut
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            # First block uses stride 2
            layers.append(ResidualBlock(in_channels, out_channels, stride=2, dropout_prob=dropout_prob))
            in_channels = out_channels
        else:
            shortcut = nn.Identity()
            layers.append(ResidualBlock(in_channels, out_channels, stride=1, dropout_prob=dropout_prob))
            in_channels = out_channels
            
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels, stride=1, dropout_prob=dropout_prob))
            
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
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
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super().__init__()
        # Block uses 3x3 Conv2d, norm, activation, optional Dropout2d
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        
        # Optional Dropout2d
        self.drop = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        
        # Shortcut
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
        out = self.act1(out)
        out = self.drop(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop(out)
        
        out += identity
        return out