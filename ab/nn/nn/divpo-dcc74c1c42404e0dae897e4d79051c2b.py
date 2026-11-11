import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = self._build_stem(in_shape, prm)
        self.classifier = nn.Linear(64, int(out_shape[0]))
        self._init_weights()

    def _build_stem(self, in_shape, prm):
        channels = in_shape[1]
        stem = nn.Sequential()
        stem.add_module('conv', nn.Conv2d(channels, 32, 3, 1, 1, bias=False))
        stem.add_module('bn', nn.BatchNorm2d(32))
        stem.add_module('relu', nn.ReLU())
        stem.add_module('pool', nn.MaxPool2d(2))
        stem.add_module('affine2', nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        stem.add_module('bn2', nn.BatchNorm2d(64))
        stem.add_module('relu2', nn.ReLU())
        return stem

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        return self.classifier(x.mean(3).mean(2))

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()