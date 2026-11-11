import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SmallBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride), 
            nn.BatchNorm2d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        orig = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.downsample(orig)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.first_conv = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.small_block_1 = self._make_small_block_layer(32, 32)
        self.small_block_2 = self._make_small_block_layer(32, 32, stride=2)
        self.small_block_3 = self._make_small_block_layer(32, 32, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.features = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(64 * 6 * 6, self.num_classes)

    def _make_small_block_layer(self, in_ch, out_ch, stride=1):
        layer = []
        layer += [SmallBlock(in_ch, out_ch, stride)]
        for i in range(1, stride):
            layer += [SmallBlock(out_ch, out_ch, 1)]
        return nn.Sequential(*layer)

    def features_forward(self, x):
        x = self.first_conv(x)
        x = self.small_block_1(x)
        x = self.small_block_2(x)
        x = self.small_block_3(x)
        return x

    def forward(self, x):
        x = self.features_forward(x)
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (6, 6))
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=prm['lr'], momentum=prm['momentum'],
            weight_decay=1e-4, dampening=0, nesterov=False
        )

    def learn(self, train_data):
        self.train()
        for data, tar in train_data:
            data, tar = data.to(self.device), tar.to(self.device)
            self.optimizer.zero_grad()
            y = self(data)
            loss = self.criteria(y, tar)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()