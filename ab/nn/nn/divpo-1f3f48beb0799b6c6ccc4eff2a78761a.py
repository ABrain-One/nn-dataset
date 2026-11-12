import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:
            in_channels = in_shape[1]
        else:
            in_channels = int(in_shape[0])
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.stem.weight.data.normal_(0, 0.1)
        self.block1 = Block(32, 32, True)
        self.block2 = Block(32, 64, True)
        self.block3 = Block(64, 64, True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block2(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
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
    def __init__(self, in_channels, out_channels, downsample=False):
        super(Block, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.ideep = int(out_channels / 2)
        self.rbase = int(out_channels / 8) + 3

    def zero_grad_slow(self, v):
        if v.grad is not None:
            v.grad.detach_()
            v.grad.zero_()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


# Architecture hints:
# Weight <= 500K; block size <= 17 (consider < 16 for best edge-perf)
# Edge-friendly: latency, edge-friendly blocks; 1.4s/300-750 < 200ms.
# Pick GELU/RELU3/RELU6: see e-friendly blocks below; no CS/TS/POW!
# Simpler v1/v2 (no in1x/in6, No SE, No DConv; < 500K).
# Resource CAREFULLY to hit < 500K; plan for blocks under 17.6x6.

# Structure guidelines:
#   - 1.3K-2.5K/block<=17, e.g.
#         - 15x (v2 or v1, no SE, 2-3 expanders)
#         -  6x   (1.1b; 1.0+3-5.5x SE, 0.2x no_in)
#         - 21.6x (2.2-4.2b w/ SE, 0.1-0.3x no_in)
#   - 1.4-2.5M/300-750: slow/fast_forward(0.8, 0.9)
#   - 50%+ p1_GELU|p2_RELU3|p3_RELU6 (RELU3=20% slower)
#   -  8-10x.5_SE(3)_d_conv<=4 (11-22% slow)
#   - 0.3-0.6% dropout<=0.5 (edge-friendly)
#   -  2-4x   expander(2-4, .45-1.00x)
#   -  1.6-2.0x   (1.4-2.0x=1.6+0.2x SE)

# Analysis:
#   -   8.3/10.5   (v1.1/.9)
#   -  12.2/16.2   (v1.3/.8)
#   -   7.5/9.0    (v1.1/.9, 1.0+3.5x SE)
#   -   3.7/4.7    (v1.0)
#   -  14.3/17.5   (v2/1.1)
#   -   3.7/4.7    (v2/1.0)
#   -   7.9/9.3    (v2/1.0 + 2.4x SE)
#   -   8.3/10.1   (v1.3/.9, 1.0+2.7x SE)
#   -  12.0/14.3   (v2/1.2)
#   -   7.5/9.0    (v1.1/.9, 1.0+3.5x SE)
#   -   8.3/10.5   (v1.1/.9, 1.0+3.5x SE)
#   -   7.5/9.0    (v1.1/.9, 1.0+3.5x SE)

#   Edge-friendly blocks (pdrop=0.5):
#     - 1.2x  v1_3x3   (1.0+3x SE)
#     - 1.4x  v1_3x3   (1.0+3x SE)
#     - 1.4x  v2_3x3   (1.0+3x SE)