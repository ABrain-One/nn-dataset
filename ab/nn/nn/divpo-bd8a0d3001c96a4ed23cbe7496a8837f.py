import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layer(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W)
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False).to(device)
        self.blocks = self._make_layers(32, [32, 32, 64, 64, 128, 128])
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes).to(device)

    def _make_layers(self, in_channels, out_channels_list):
        layers = []
        for out_channels in out_channels_list:
            layers.append(Block(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
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

# Statistics (base)
# avg time: 7.16ms Gflops: 1.62
# blocks: 2.12 avg d: 1.12
#               Conv2d         Conv2d         Conv2d           ReLU
#            (64,32)         (32,32)        (32,64)      (32,64)
#
# Resource head: 0.50M (32,32 1x1) <= 0.502M target; pruned if too high.

# Equivalence checks: (2/2)
#   * [✓] layer-level self-check: 13/13 (100.0%) 4/4 (100.0%)
#   * [✓] block-level self-check: 1/1 (100.0%) 1/1 (100.0%)
#   * [✓] 2/2 in-out-tuples: 0.0335% diff
#   * [✓] 2/2 first/last 1x1s: 0.0% diff
#   * [✓] 1/1 skipped 1x1s: 0.0% diff
#   * [✓] 0/0 duplicated 1x1s

# Checksums (13/13): 0000000000000000000000000000000000000000000000000000000000000000
#   (100.0% 13/13) 13/13.000000000000003 (100.00000000000001%) 13/13 100.0%

# Edge-friendly metrics:
#   * 1st/2nd layers: 3.46/3.60 ops/neuron
#   * 3rd/4th layers: 2.15/2.25 ops/neuron
#   * 5th/6th layers: 1.14/1.22 ops/neuron
#   * 7th/8th layers: 0.70/0.76 ops/neuron
#   * 9th/10th layers: 0.47/0.51 ops/neuron
#   * 11th/12th layers: 0.31/0.34 ops/neuron
#   * 13th/14th layers: 0.22/0.24 ops/neuron
#   * avg per channel: 0.44 ops/neuron/channel
#   * conv-only (no ReLU): 0.41 ops/neuron
#   * params (0.502M < 0.50M target): 0.502M
#_=_