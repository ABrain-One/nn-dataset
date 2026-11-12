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
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.first_conv = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1, bias=False)
        self.bn_first_conv = nn.BatchNorm2d(self.features)
        self.ac_first_conv = nn.ReLU(inplace=True)
        self.second_group = self._make_layer(32, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers += [self._make_layer_block(self.features, out_channels, stride)]
        for i in range(1, blocks):
            layers += [self._make_layer_block(out_channels, out_channels)]
        return nn.Sequential(*layers)

    def _make_layer_block(self, in_channels, out_channels, stride=1):
        layer_block = nn.Sequential()
        layer_block.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layer_block.add_module('bn1', nn.BatchNorm2d(out_channels))
        layer_block.add_module('relu', nn.ReLU(inplace=True))
        layer_block.add_module('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layer_block.add_module('bn2', nn.BatchNorm2d(out_channels))
        return layer_block

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bn_first_conv(x)
        x = self.ac_first_conv(x)
        x = self.second_group(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=1e-4)
        
    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3.0)
            self.optimizer.step()