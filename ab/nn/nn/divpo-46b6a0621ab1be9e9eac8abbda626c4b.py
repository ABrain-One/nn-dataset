import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self._make_layer(32, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.2))

    def _make_layer(self, inplanes, planes, blocks, stride):
        layers = [nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        for _ in range(1, blocks):
            layers += [nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                       nn.BatchNorm2d(planes),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def __repr__(self):
        return f'Net(blocks=8, channels=[32, 64, 128], channels_out=[64, 128, 256])'

    def to(self, device):
        self.device = device
        return super(Net, self).to(device)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.to('cpu')
        self.device = None
        self.stem = self.stem.to('cpu')
        self.layer1 = self.layer1.to('cpu')
        self.layer2 = self.layer2.to('cpu')
        self.avgpool = self.avgpool.to('cpu')
        self.classifier = self.classifier.to('cpu')
        self.dropout = self.dropout.to('cpu')
        del self.criteria
        del self.optimizer
        torch.cuda.empty_cache()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x