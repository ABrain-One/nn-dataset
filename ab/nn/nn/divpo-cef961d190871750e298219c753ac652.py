import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 64)
        self.block5 = Block(64, 64)
        self.block6_1 = Block(64, 64)
        self.block6_2 = Block(64, 64)
        self._avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flat_to_class = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.3)),
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
        x = self.block2(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block3(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block4(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block5(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block6_1(x)
        x = self.block6_2(x)
        x = self._avgpool(x)
        x = x.flatten(1)
        x = self.flat_to_class(x)
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

# Optional: helper blocks AFTER Net definition
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.act(x)
        return x