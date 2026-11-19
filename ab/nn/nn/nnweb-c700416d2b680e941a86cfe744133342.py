import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class CIFARResidualNet(nn.Module):
    def __init__(self, in_channels, num_classes, prm):
        super(CIFARResidualNet, self).__init__()
        self.prm = prm
        dropout_p = prm['dropout']
        
        # Stem: 3x3 Conv2d, norm, activation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: residual blocks at 32x32
        self.stage1 = self._make_stage(64, 64, num_blocks=2, stride=1, dropout_p=dropout_p)
        
        # Stage 2: residual blocks at 16x16
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2, dropout_p=dropout_p)
        
        # Stage 3: residual blocks at 8x8
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2, dropout_p=dropout_p)
        
        # Ends with GAP and classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)
        
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, dropout_p):
        layers = []
        for i in range(num_blocks):
            stride_i = stride if i == 0 else 1
            layers.append(ResidualBlock(in_channels if i == 0 else out_channels, out_channels, stride_i, dropout_p))
            in_channels = out_channels
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
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout_p):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Optional Dropout2d
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

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
        class_number = out_shape[0]

        self.model = CIFARResidualNet(channel_number, class_number, prm)

    def forward(self, x):
        return self.model(x)

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