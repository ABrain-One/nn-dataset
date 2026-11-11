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
        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.block1 = self._make_block(32, 32, 1, 1)
        self.block2 = self._make_block(32, 64, 2, 1)
        self.block3 = self._make_block(64, 64, 2, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Linear(64 * 8 * 8, self.num_classes)

    def _make_block(self, in_channels, out_channels, stride, repeat):
        block = []
        block.append(nn.ReLU(inplace=True))
        block.append(nn.BatchNorm2d(in_channels))
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        block.append(nn.ReLU(inplace=True))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0))
        block.append(self._make_group(out_channels, out_channels, repeat))
        return nn.Sequential(*block)

    def _make_group(self, in_channels, out_channels, repeat):
        groups = []
        for i in range(repeat):
            stride = 1 if i > 0 else 2
            groups.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=stride))
            groups.append(nn.ReLU(inplace=True))
            groups.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*groups)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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