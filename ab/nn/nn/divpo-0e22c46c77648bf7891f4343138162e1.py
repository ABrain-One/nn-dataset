import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1]) if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = self._make_block(32, 3)
        self.avgpool = nn.AvgPool2d(2)
        self.block2 = self._make_block(32, 3)
        self.classifier = nn.Linear(32, num_classes)

    def _make_block(self, planes, blocks):
        block = nn.Sequential()
        for _ in range(blocks):
            block.add_module(f"conv1", nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False))
            block.add_module(f"bn1", nn.BatchNorm2d(planes))
            block.add_module(f"relu", nn.ReLU(inplace=True))
        return block

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.avgpool(x)
        x = self.block2(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
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