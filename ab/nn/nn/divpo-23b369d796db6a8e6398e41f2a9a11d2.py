import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SmallBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(SmallBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.ReLU(inplace=False)

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        self.bn2(out)
        out += shortcut
        out = self.act2(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.first_conv = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.small_block_1 = self.build_small_block(32, 32)
        self.small_block_2 = self.build_small_block(32, 32)
        self.small_block_3 = self.build_small_block(32, 32)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, self.num_classes)

        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        x = self.first_conv(x)
        x = self.small_block_1(x)
        return self.classifier(self.avg_pool(x).flatten(1))

    def build_small_block(self, in_channels: int, out_channels: int) -> SmallBlock:
        return SmallBlock(in_channels)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

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