import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_path=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.drop_path = nn.Identity() if drop_path == 0.0 else DropPath(drop_path)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        if self.scale_by_keep:
            random_tensor.div_(keep_prob)
        random_tensor.floor_()  # binarize
        return x * random_tensor

class ResNetFeatures(nn.Module):
    def __init__(self, in_shape, out_shape, prm):
        super(ResNetFeatures, self).__init__()
        
        # Initialize stochastic depth rates BEFORE creating layers
        # Configuration: 4 stages, 2 blocks each = 8 blocks total
        total_blocks = 8
        # Linear schedule from 0 to 0.2 (max drop rate)
        max_drop_rate = 0.2
        self.drop_rates = [x / total_blocks * max_drop_rate for x in range(total_blocks)]
        self.current_block_idx = 0
        
        # Small ResNet-18 config: 32, 64, 128, 256
        # This keeps param count well under 5M (~1.5M)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            # Apply drop path rate based on current global block index
            drop_rate = self.drop_rates[self.current_block_idx]
            self.current_block_idx += 1
            
            layers.append(BasicBlock(in_planes, planes, stride, drop_path=drop_rate))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.prm = prm
        
        self.features = ResNetFeatures(in_shape, out_shape, prm)
        self.num_classes = out_shape[0] if out_shape else 10
        
        # Use dropout from prm
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.0))
        
        # Final feature dim is 256 (from last layer channels)
        self.classifier = nn.Linear(256, self.num_classes)
        
        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm['momentum'], 0.999), weight_decay=1e-4)
        self.max_batches = prm.get('max_batches', None)
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            if self.max_batches is not None and i >= self.max_batches: break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()