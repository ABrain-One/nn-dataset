import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, dropout_p=0.0, use_se=False, use_stochastic_depth=False, drop_prob=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_p) if use_dropout else nn.Identity()
        self.stochastic_depth_prob = drop_prob
        self.use_stochastic_depth = use_stochastic_depth
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.se = None
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, max(1, out_channels // 16)),
                nn.ReLU(inplace=True),
                nn.Linear(max(1, out_channels // 16), out_channels),
                nn.Sigmoid()
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.dropout(out)
        
        if self.use_stochastic_depth and self.training:
            if torch.rand(()) < self.stochastic_depth_prob:
                out = torch.zeros_like(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.se is not None:
            se_out = self.se(out)
            out = out * se_out
        
        out += identity
        out = self.relu(out)
        
        return out


class CIFARResidualNet(nn.Module):
    def __init__(self, in_channels, num_classes, depth=20, width=32, use_dropout=False, dropout_p=0.0, use_se=False, use_stochastic_depth=False, drop_prob=0.0):
        super(CIFARResidualNet, self).__init__()
        
        # Calculate number of blocks per stage
        # Standard ResNet-20 for CIFAR: 3 stages, (20-2)/6 = 3 blocks per stage
        # But the description says "Stage 1: residual blocks at 32x32, Stage 2: ... 16x16, Stage 3: ... 8x8"
        # It doesn't specify depth explicitly, but "custom small residual network" suggests a standard small size like ResNet-20 or similar.
        # Let's assume depth=20 for a "small" network, which gives 3 blocks per stage.
        # If depth is not a multiple of 6, we adjust.
        if (depth - 2) % 6 != 0:
            raise ValueError("Depth should be 2 + 6n")
        num_blocks_per_stage = (depth - 2) // 6
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 32x32
        self.stage1 = self._make_stage(width, width, num_blocks_per_stage, stride=1, use_dropout=use_dropout, dropout_p=dropout_p, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob)
        
        # Stage 2: 16x16
        self.stage2 = self._make_stage(width, width * 2, num_blocks_per_stage, stride=2, use_dropout=use_dropout, dropout_p=dropout_p, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob)
        
        # Stage 3: 8x8
        self.stage3 = self._make_stage(width * 2, width * 4, num_blocks_per_stage, stride=2, use_dropout=use_dropout, dropout_p=dropout_p, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(width * 4, num_classes)
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride, use_dropout, dropout_p, use_se, use_stochastic_depth, drop_prob):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_dropout, dropout_p, use_se, use_stochastic_depth, drop_prob))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, use_dropout, dropout_p, use_se, use_stochastic_depth, drop_prob))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]
        
        # Initialize the CIFARResidualNet
        # Using standard choices for "small" network: Depth 20, Width 32
        # No SE, no stochastic depth, no dropout (standard baseline)
        self.model = CIFARResidualNet(
            in_channels=self.channel_number,
            num_classes=self.class_number,
            depth=20,
            width=32,
            use_dropout=False,
            dropout_p=0.0,
            use_se=False,
            use_stochastic_depth=False,
            drop_prob=0.0
        )
        
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.float()
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()