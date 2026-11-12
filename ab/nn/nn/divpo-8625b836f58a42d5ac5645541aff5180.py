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
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            self._make_layer(32, 64, 2),
            self._make_layer(64, 96, 1)
        )
        self.classifier = nn.Linear(96, self.num_classes)
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.0))

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        self.expansion = 1
        layers.append(self._make_basic_block(in_channels, out_channels, blocks))
        return nn.Sequential(*layers)

    def _make_basic_block(self, in_channels, out_channels, blocks):
        strides = 1
        layers = []
        layers.append(self._make_conv_block(in_channels, out_channels, strides))
        layers += [self._make_conv_block(out_channels, out_channels, stride)
                   for stride in [1] * (blocks - 1)]
        return nn.Sequential(*layers)

    def _make_conv_block(self, in_channels, out_channels, stride):
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=stride, padding=padding),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=(2, 3))
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