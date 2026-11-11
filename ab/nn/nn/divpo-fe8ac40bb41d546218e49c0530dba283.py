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
        self.stem = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)
        self.features = nn.Sequential(
            self._make_block(16, 16, 5, 2),
            self._make_block(16, 32, 3, 2),
            self._make_block(32, 32, 3, 2),
            self._make_block(32, 64, 3, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def _make_block(self, in_channels, out_channels, kernel_size, stride):
        block = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True)]
        in_channels = out_channels
        while True:
            if self._should_add_skip_connection(in_channels, out_channels, stride):
                block.append(nn.Conv2d(in_channels, out_channels, 1, bias=False))
                block.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=1, bias=False))
            block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.ReLU(inplace=True))
            if self._should_add_skip_connection(in_channels, out_channels, stride):
                break
            stride = 1
        return nn.Sequential(*block)

    def _should_add_skip_connection(self, in_channels, out_channels, stride):
        return in_channels != out_channels or stride != 1

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