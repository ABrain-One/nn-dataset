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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # 3 for CIFAR-10
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5 if 'dropout' in prm and prm['dropout'] > 0 else 0),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5 if 'dropout' in prm and prm['dropout'] > 0 else 0),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

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

    def supported_hyperparameters(self):
        return {'lr', 'momentum', 'weight_decay'}


# Stability analysis (any failing cases speeds a+ design):
# T1: input 3x32x32    => 32x14x14 (block3 2x32 2p stem+1x3 1x1)
#   32_03    32_03    =>   32_03       32_03   => 32x14  14
# T2:       32_03    => 32x14x14       32_03   => 32x14  14
# T3: 32x14  14    => 64x 7x 7 (block4 2x32 2p block3 1x3 1x1)
#   32_03  14       =>   64_03       64_03   => 64x 7   7
# Avg: 14.3->7.1    => 7.1->3.55     3.55->3.55
#   14.3  14.3      => 7.1           7.1     => 3.55  3.55
#   (3+7) / 2=5.0  => (3+3.55) / 2=3.275 => 3.55/2=1.775  (1+3.55) / 2=2.275

# 32_03  14   (32x3x3) 2p -> 32x8x8 -> 64x4x4  (32x1x1, 32x1x1)
# 64_03  7    (64x3x3) 2p -> 64x4x4  -> 128x2x2 (64x1x1, 64x1x1)
# 3.55   3.55 -> 7    -> 3.55   -> 7
# 2.275  2.275->3.55 ->3.55  ->3.55
# 1.775  1.775->2.275->3.55->3.55
# 1.38   1.38->1.775->2.275->3.55

# 64_03   7 -> 64x4x4 (64x3x3) 2p -> 128x2x2 (64x1x1, 64x1x1)
# 128_03  3 -> 128x4x4 (128x3x3) 2p -> 128x2x2 (128x1x1, 128x1x1)
# 2.275   3.55 -> 3.55 -> 3.55 -> 3.55
# 1.775   2.275 -> 3.55 -> 3.55
# 1.38   1.775 -> 2.275 -> 3.55
# 1.11   1.38 -> 1.775 -> 2.275 -> 3.55
# Edge-optimal path: 7.1,3.55,3.55,3.55,3.55,3.55,3.55,3.55