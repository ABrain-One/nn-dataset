import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.features = 32
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.first_conv = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1)
        self.features = 32
        self.layer1 = self._make_layer(32, 32, 1)
        self.layer2 = self._make_layer(32, 64, 2)
        self.layer3 = self._make_layer(64, 128, 2)
        self.layer4 = self._make_layer(128, 256, 2)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, self.num_classes)

    def _make_layer(self, in_channels, out_channels, blocks_num):
        block = nn.Sequential(
            self._building_block(in_channels, out_channels, bulk=blocks_num)
        )
        return block

    def _building_block(self, in_channels, out_to, bulk=1):
        blocks = []
        for building_block in range(bulk):
            stride = 1
            if in_channels != out_to and building_block == 0:
                stride = 2
            blocks += [
                nn.Conv2d(in_channels, out_to, kernel_size=3,
                          padding=1, stride=stride),
                nn.BatchNorm2d(out_to),
                nn.ReLU(),
                nn.Conv2d(out_to, out_to, kernel_size=3,
                          padding=1),
                nn.BatchNorm2d(out_to),
                nn.ReLU()
            ]
            in_channels = out_to
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.glob_avg_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
            weight_decay=1e-4
        )

    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
            self.optimizer.step()