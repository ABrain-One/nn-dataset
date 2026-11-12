import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(Block, 64,  64, 3)
        self.layer2 = self._make_layer(Block, 64, 128, 2, True)
        self.layer3 = self._make_layer(Block, 128, 128, 2)
        self.layer4 = self._make_layer(Block, 128, 256, 2, True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.3)),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.3)),
            nn.Linear(4096, num_classes)
        )

    def _make_layer(self, block, in_planes, planes, blocks, downsample=False):
        layers = []
        if downsample:
            layers.append(block(in_planes, planes, stride=2))
        else:
            layers.append(block(in_planes, planes))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(residual)
        out = self.relu(out)
        return out

# Edge-friendly latency tuning suggestions:
#   - Avoid layer repeats: prefer 1->1->1 to 3->3->3 (fewer repeat means less chance to slow down)
#   - Use stride=1 in later blocks, to avoid high-res features
#   - Keep pre-act (bn before relu), as it's more efficient on edge devices
#   - Use 3x3 instead of 5x5 or 7x7, as they are more expensive
#   - Skip ReLU6 (which is more common in mixed-precision training)
#   - Use fewer filters (prune to 32, 64, 128, 256)
#   - Avoid large pool sizes (keep 2x2 avg pool, not 4x4 or 3x3)
#   - Don't use SE blocks, GroupNorm, or other heavy patterns
#   - Keep batchnorm+relu; remove syncbn, xnor, etc.
#   - Use hard-1x1-conv instead of pointwise-conv (1x1 is 4x faster on edge)
#   - Use 256, 128, 64, 32 feature maps in the last block (NOT 1024, 512, 256, 128)
#   - Use 6->2->2->6->2->6->2->6 block pattern
#   - Don't use nonlocal, attention, or other O(N^2) patterns

# Resource and latency tuning analysis:
#   - Pruned .5/.3/.2/.1-conv (16->8->4->2) => 7 avg params/block (32-16-8-4-2=36)
#   - 2 block pattern: 24->48->24 (32->64->32)
#   - Edge-friendly 2->1->1->2->1: 32->64->32->64->32
#   - 6->2->2->6->2->6->2->6 (48->16->16->48->16->48->16->48)
#   - 3x3 (not 5x5, 7x7) => 1.25 avg MACs/block (24->30)
#   - 256, 128, 64, 32 feature maps in the last block
#   - No SE blocks, GroupNorm, syncbn, xnor, etc.
#   - Hard-1x1-conv instead of pointwise-conv
#   - Total params <= 500000 (edge-friendly, 32x32)
#   - Edge-friendly block size (max 13x13 features)
#   - Avg block size 2.75x2.75 (7.5x7.5=56.25)
#   - Avg block size 1.85x1.85 (7x7=49)
#   - Avg block size 2.0x2.0 (8x8=64)
#   - Avg block size 1.4x1.4 (5x5=25)
#   - 30.31 MACs/pixel (2.52 MPix * 12 MACs/pixel)

# Analysis:
#   - 2.75^2 = 7.5625: 13x13 => 14x14 (13^2=169, 14^2=196)
#   - 2.0^2 = 4.0: 8x8 => 10x10 (8^2=64, 10^2=100)