import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16, dropout=0.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.0, use_se=True, drop_path_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes, reduction=16, dropout=dropout)
        else:
            self.se = nn.Identity()

        self.drop_path_rate = drop_path_rate

        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.dropout(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.drop_path_rate > 0.0:
            # Stochastic depth
            if self.training:
                keep_prob = 1.0 - self.drop_path_rate
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
                out = out * mask / keep_prob
            # else: identity

        out += identity
        out = self.relu(out)
        return out


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.dropout = prm['dropout']
        
        # CIFAR-10 is 32x32. 
        # Stem: 3x3 Conv, Norm, Act. Output 32x32.
        # Stage 1: Residual blocks at 32x32.
        # Stage 2: Residual blocks at 16x16.
        # Stage 3: Residual blocks at 8x8.
        
        # Standard choice for "small" ResNet on CIFAR: ResNet-20 or similar.
        # Let's use ResNet-20 structure: 3 stages, 2 blocks each.
        # Widths: 64, 128, 256.
        
        num_blocks_per_stage = 2
        widths = [64, 128, 256]
        
        self.in_planes = widths[0]
        
        # Stem
        self.conv1 = nn.Conv2d(in_shape[1], self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1: 32x32
        self.layer1 = self._make_layer(widths[0], num_blocks_per_stage, stride=1)
        
        # Stage 2: 16x16
        self.layer2 = self._make_layer(widths[1], num_blocks_per_stage, stride=2)
        
        # Stage 3: 8x8
        self.layer3 = self._make_layer(widths[2], num_blocks_per_stage, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        # Standard choice: Linear layer. Dropout is applied in blocks.
        # We can also add dropout before the linear head if desired, but the prompt says "Blocks use ... optional Dropout2d".
        # The hyperparameter 'dropout' must be used. It is used in the blocks.
        # To be safe and ensure it's "actually used" in a way that affects the final prediction path if blocks didn't have it (though they do),
        # we'll just rely on the block dropout.
        
        self.classifier = nn.Linear(widths[2], out_shape[0])

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, dropout=self.dropout, use_se=True, drop_path_rate=0.0))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out