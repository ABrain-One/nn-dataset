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
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.block1 = self._block(64, 64, 3, 1)
        self.block2 = self._block(64, 128, 3, 2)
        self.block3 = self._block(128, 128, 3, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.dropout = nn.Dropout(prm.get('dropout', 0.3))
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _block(self, in_channels, out_channels, kernel_size, stride):
        block = nn.ModuleList()
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU())
        block.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU())
        block.append(nn.AvgPool2d(2))
        return block

    def forward(self, x):
        x = self.act2(self.bn2(self.conv1(self.act1(self.bn1(self.stem(x))))))
        for layer in self.block1:
            x = layer(x)
        for layer in self.block2:
            x = layer(x)
        for layer in self.block3:
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc2(self.act1(self.fc1(x)))
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