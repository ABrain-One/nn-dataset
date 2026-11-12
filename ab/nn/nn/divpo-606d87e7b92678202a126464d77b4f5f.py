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
        
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.stem.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)
        
        self.features = nn.Sequential(
            self._make_block(32, 32, 3, 1),
            self._make_block(32, 64, 3, 2),
            self._make_block(64, 64, 3, 3)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)

    def _make_block(self, in_channels, out_channels, kernel_size, num_blocks):
        blocks = [Block(in_channels, out_channels, kernel_size)]
        for _ in range(1, num_blocks):
            blocks.append(Block(out_channels, out_channels, kernel_size))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = self._shortcut(in_channels, out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def _shortcut(self, in_channels, out_channels):
        if in_channels != out_channels:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(identity)
        return self.act2(x)