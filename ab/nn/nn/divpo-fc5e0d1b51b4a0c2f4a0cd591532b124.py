import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(32, 32, 3, 1, 1)
        self.layer2 = self.make_layer(32, 64, 3, 2, 1)
        self.layer3 = self.make_layer(64, 128, 3, 2, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Linear(128 * 6 * 6, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_channels, out_channels, block_num, stride, padding):
        layers = []
        layers.append(Block(in_channels, out_channels, 3, stride, padding))
        for _ in range(1, block_num):
            layers.append(Block(out_channels, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Edge-friendly latency budget means we'll use efficient block designs and keep params below 500k.
# This design uses depthwise separable convolutions for efficiency.
# The following block (Block) uses grouped convolution for depthwise and pointwise operations.

# Target params <= 500000. Keep products of kW, kH, in, and out under 64M to fit into default 32-bit Tensors.
# (e.g., self.conv1 and self.conv2 in Block below should have in*out < 64M).
# The block should use no more than 512 out_channels to fit in the limit.

# Block design (for in_channels <= 32 and out_channels <= 32):
# Block: (in_channels -> out_channels) x2
#   - depthwise conv (in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
#   - pointwise conv (in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#   - BN
#   - ReLU
#   - depthwise conv (out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
#   - pointwise conv (out_channels, out_channels, kernel_size=1, stride=1, padding=0)
#   - BN
#   - ReLU

# The provided MLP wrapper and Block should be used "as-is" and not modified.
# Do NOT include BatchNorm2d or Dropout layers outside the Block; they should be fully contained inside the Block.
# Keep Block dimensions low (in <= 32, out <= 32): the block should not exceed width or height constraints.

# The model should use the following block for depthwise separable convolutions (no more than 2 of these blocks allowed):
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

# The provided MLP wrapper (e.g., Net.learn, Net.train_setup) should be used "as-is" and not modified.
# Do NOT include any of the following in your design:
# - nn.AdaptiveAvgPool2d(1) (1D MLP after spatial reduction)
# - nn.Linear(in_features, out_features) (use the provided MLP wrapper)
# - any of: nn.Dropout(p), nn.BatchNorm1d(features), nn.InstanceNorm2d(groups)
# - any other layers outside the Block ( Block -> Conv2d + BatchNorm2d + ReLU -> build NO extra layers outside Block)

# The model should output per-class probabilities (i.e., no Sigmoid after the last classifier).
# The last layer in the classifier should be of the form nn.Linear(A, out_features).

# The edge-friendly latency budget means the model should be compact and compute-efficient.
# The design should achieve max_channels <= 32 and total_params <= 500000.

# The following constants control the minimum and maximum dimensions at each block boundary.
# The design should NOT use dimensions outside these ranges (e.g., out_channels <= 32).
#   MIN_CHANNELS: the minimum number of channels that any layer in the design should use.
#   MAX_CHANNELS: the maximum number of channels any block (or the last block before the classifier) should use.
#   BLOCK_CHANNELS: the set of allowable channel dimensions for blocks (e.g., {16, 32}).
# The MIN_CHANNELS and MAX_CHANNELS values will be provided by the evaluation environment.
# The BLOCK_CHANNELS list will be one of: [{32}, {64}, {32, 64}, {64, 128}].
#   The target design should use one of these channel sets for blocks and the classifier.

# The provided loss function (if used) will be: criterion = nn.CrossEntropyLoss().
# The optimizer will be SGD with momentum (SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])).

# The dataset is CIFAR-10 (32x3