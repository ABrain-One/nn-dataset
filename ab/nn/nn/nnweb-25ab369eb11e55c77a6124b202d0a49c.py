import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, use_se=False, use_stochastic_depth=False, drop_prob=0.0, se_reduction=16, stochastic_depth_prob=0.0):
        super(ResidualBlock, self).__init__()
        self.use_dropout = use_dropout
        self.use_se = use_se
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_prob = stochastic_depth_prob

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=drop_prob)
        else:
            self.dropout = nn.Identity()

        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, max(1, out_channels // se_reduction)),
                nn.ReLU(inplace=True),
                nn.Linear(max(1, out_channels // se_reduction), out_channels),
                nn.Sigmoid()
            )
        else:
            self.se = nn.Identity()

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

        if self.use_se:
            se_out = self.se(out)
            out = out * se_out

        if self.use_stochastic_depth and self.training:
            if torch.rand(1).item() < self.stochastic_depth_prob:
                out = 0.0
            else:
                out = out / (1.0 - self.stochastic_depth_prob)

        out += identity
        out = self.relu(out)
        return out


class CIFARResidualNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, depth=20, base_width=32, use_dropout=False, use_se=False, use_stochastic_depth=False, drop_prob=0.0, se_reduction=16, stochastic_depth_prob=0.0):
        super(CIFARResidualNet, self).__init__()

        # Stem: 3x3 Conv -> Norm -> Activation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        # Determine number of blocks per stage based on depth
        # Standard ResNet-20 has 3 stages with 3 blocks each (3*2 + 2 = 8 layers, plus stem and head).
        # Depth formula: 6*n + 2. For depth=20, n=3.
        n = (depth - 2) // 6
        if n < 1:
            n = 1

        # Stage 1: 32x32
        self.stage1 = nn.Sequential(
            ResidualBlock(base_width, base_width, stride=1, use_dropout=use_dropout, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob, se_reduction=se_reduction, stochastic_depth_prob=stochastic_depth_prob),
            ResidualBlock(base_width, base_width, stride=1, use_dropout=use_dropout, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob, se_reduction=se_reduction, stochastic_depth_prob=stochastic_depth_prob),
            ResidualBlock(base_width, base_width, stride=1, use_dropout=use_dropout, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob, se_reduction=se_reduction, stochastic_depth_prob=stochastic_depth_prob)
        )

        # Stage 2: 16x16
        self.stage2 = nn.Sequential(
            ResidualBlock(base_width, base_width * 2, stride=2, use_dropout=use_dropout, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob, se_reduction=se_reduction, stochastic_depth_prob=stochastic_depth_prob),
            ResidualBlock(base_width * 2, base_width * 2, stride=1, use_dropout=use_dropout, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob, se_reduction=se_reduction, stochastic_depth_prob=stochastic_depth_prob),
            ResidualBlock(base_width * 2, base_width * 2, stride=1, use_dropout=use_dropout, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob, se_reduction=se_reduction, stochastic_depth_prob=stochastic_depth_prob)
        )

        # Stage 3: 8x8
        self.stage3 = nn.Sequential(
            ResidualBlock(base_width * 2, base_width * 4, stride=2, use_dropout=use_dropout, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob, se_reduction=se_reduction, stochastic_depth_prob=stochastic_depth_prob),
            ResidualBlock(base_width * 4, base_width * 4, stride=1, use_dropout=use_dropout, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob, se_reduction=se_reduction, stochastic_depth_prob=stochastic_depth_prob),
            ResidualBlock(base_width * 4, base_width * 4, stride=1, use_dropout=use_dropout, use_se=use_se, use_stochastic_depth=use_stochastic_depth, drop_prob=drop_prob, se_reduction=se_reduction, stochastic_depth_prob=stochastic_depth_prob)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_width * 4, num_classes)

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
        
        # Choose standard ResNet-20 depth for CIFAR-10
        self.model = CIFARResidualNet(
            in_channels=self.channel_number,
            num_classes=self.class_number,
            depth=20,
            base_width=32,
            use_dropout=False,
            use_se=False,
            use_stochastic_depth=False
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
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()