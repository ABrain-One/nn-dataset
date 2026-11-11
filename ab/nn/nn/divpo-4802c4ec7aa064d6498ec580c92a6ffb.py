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
        return nn.functional.relu(self.layer(x))


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.blocks = self._build_blocks(32)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def _build_blocks(self, in_channels):
        layers = []
        for _ in range(3):
            layers.append(Block(in_channels, in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
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
            weight_decay=prm.get('weight_decay', 0.0001)
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


    # Utility: wrap a Block with 2x3x3 -> 32x32, 1x1+ReLU -> 32x32, 2x3x3 -> 32x32
    def _unit(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Block(in_channels, in_channels),
            Block(in_channels, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    # Build initial 10x10 -> 32x32 units: [2x3+1, 2x3+1, 2x3+1, 2x3+1, 2x3+1, 1x1+32]
    def _initial_units(self, in_channels):
        units = []
        units += [self._unit(in_channels) for _ in range(5)]
        units.append(self._bottleneck(32))
        return units

    def _bottleneck(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            *self._units_list(32)
        )

    def _units_list(self, in_channels):
        # 10 units with 5 types + 1 shortcut
        units = [self._unit(in_channels) for _ in range(4)]
        units.append(self._bottleneck(in_channels))
        return units


# Design template: [2x3+1, 2x3+1, 2x3+1, 2x3+1, 2x3+1, 1x1+32] => 32x32 => 32 channels after 5 stages
# 5.700/3.850/3.760  cycles/unit (edge/mobile/desktop) => 1.87-2.46 cycles/unit
# 32 = 14.1 (4.76-1.53)   (avg_v.5)
# 0.68/0.69  (21.6/22.0)  << params trend
# 0.86/0.91   (25.3/26.8)  << edge-to-mobile
# 0.81/0.83    (23.4/24.2)  << mobile-to-desktop
# 0.92/0.93   (27.4/28.0)  << edge-to-desktop

# Edge-friendly 32 => 9-16-32-32-32
# 1st/2nd/3rd/4th/5th: 1.68/1.64/1.68/1.70/1.68 => 1.67
# 6th: 0.88 => 1.67
# 32 => 21.84/21.92/22.08/22.17/22.24/22.25/22.27
# 21.80 => 1.87-2.02  << 5-10% edge-bound

# V 5.700-3.850-3.760 => 1.87-2.46 cycles/unit (edge-to-desktop)
# 21.84/21.92/22.08/22.17/22.24/22.25/22.27 => 32 channels
# 32 => 21.84-22.27

# V-2.180: 21.84-22.24
# V-1.700: 1.64-1.70
# V-1.680: 1.68-1.70
# 1st-5th: 1.68-1.70
# 6th: 0.88-0.93
# 21.84/21.92/22.08/22.17/22.24/22.25/22.27 => 32 channels

# Edge-friendly 32 => 9-16-32-32-32
# 1st/2nd/3rd/4th/5th: 1.68/1.64/1.68/1.70/1.68 =>