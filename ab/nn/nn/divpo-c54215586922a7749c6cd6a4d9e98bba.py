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
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.block1 = self._make_block(32, 32, 3)
        self.block2 = self._make_block(32, 64, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        dropout = prm.get('dropout', 0.3)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _make_block(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(out_channels)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(0.3) if i > 0 else nn.Identity()]
            in_channels = out_channels
        layers += [nn.MaxPool2d(2, 2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
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