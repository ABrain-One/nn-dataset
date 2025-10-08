import torch
from torch import nn

def supported_hyperparameters():
    return ['lr', 'momentum', 'dropout']


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
        self.learning_rate = prm['lr'] if 'lr' in prm else 0.01
        self.batch_size = prm['batch'] if 'batch' in prm else 64
        self.dropout_value = prm['dropout'] if 'dropout' in prm else 0.5
        self.momentum = prm['momentum'] if 'momentum' in prm else 0.9
        self.transform = prm['transform'] if 'transform' in prm else None

        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_value),
            nn.Linear(256 * 7 * 7, 128),
            nn.ReLU(True),
            nn.Linear(128, self.out_shape[0])
        )
        self.criteria = nn.CrossEntropyLoss()
    
    def _make_layers(self):
        layers = []
        in_channels = 3
        cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = int(v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        self._to_linear = None
        self._from_linear = None
                                                                         
        self._last_channels = 256
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()