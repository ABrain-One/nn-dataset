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
        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False)
        self._init_params(self.stem)
        self.blocks = nn.Sequential()
        self._add_block(32, 32, 1, 2)
        self._add_block(32, 64, 1, 2)
        self._add_block(64, 64, 1, 1)
        self.classifier = nn.Linear(64, self.num_classes, bias=False)
        self._init_params(self.classifier)
        self.prm = prm

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def _add_block(self, in_channels, out_channels, stride, groups):
        block = nn.Sequential()
        block.add_module('nn_conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False))
        self._init_params(block.nn_conv)
        block.add_module('nn_bn', nn.BatchNorm2d(out_channels))
        block.add_module('nn_relu', nn.ReLU())
        self.blocks.add_module(f'trace_{len(self.blocks)}', block)
        self._add_pooling(out_channels, stride)

    def _add_pooling(self, out_channels, stride):
        if stride > 1:
            pool = nn.MaxPool2d(2, 2)
            self.blocks.add_module(f'trace_{len(self.blocks)}', pool)

    def _init_params(self, module):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)