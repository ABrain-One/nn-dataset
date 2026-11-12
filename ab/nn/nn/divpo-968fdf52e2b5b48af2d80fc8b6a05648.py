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
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = nn.Sequential(
            self._make_layer(32, 3, 1),
            self._make_layer(64, 3, 2),
            self._make_layer(128, 3, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(128 * 7 * 7, self.num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layer = []
        layer += [nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)]
        layer += [nn.BatchNorm2d(out_channels)]
        layer += [nn.ReLU(inplace=True)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layer += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)]
            layer += [nn.BatchNorm2d(out_channels)]
            layer += [nn.ReLU(inplace=True)]
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.learning_rate, 
            momentum=self.momentum,
            weight_decay=0.0
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