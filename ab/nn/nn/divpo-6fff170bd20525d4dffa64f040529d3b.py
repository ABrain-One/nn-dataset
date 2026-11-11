import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = self._make_block(32, 64, 32, 1)
        self.block2 = self._make_block(64, 128, 16, 2)
        self.block3 = self._make_block(128, 256, 8, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(256 * 7 * 7, num_classes)

    def _make_block(self, in_channels, out_channels, stride, repeat):
        block = []
        block.append(nn.BatchNorm2d(in_channels))
        block.append(nn.ReLU(inplace=True))
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride))
        for _ in range(repeat - 1):
            block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.ReLU(inplace=True))
            block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        return nn.Sequential(*block)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

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