import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build__layer_2(128, 256)
        self.layer5 = self.build_layer(256, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0]),
        )
        self._last_layer_name = 'classifier'
        self._last_dim = out_shape[0]
        self._last_dropout = prm.get('dropout', 0.5)
        self._last_activation = nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self._count_params()

    def build_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        for _ in range(num_blocks - 1):
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def build__layer_2(self, in_channels, out_channels):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

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

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _last_layer_name(self):
        return self._last_layer_name

    def _last_dim(self):
        return self._last_dim

    def _last_dropout(self):
        return self._last_dropout

    def _last_activation(self):
        return self._last_activation

    def _count_params(self):
        self._last_layer_name = None
        self._last_dim = None
        self._last_dropout = None
        self._last_activation = None
        count = 0
        for p in self.parameters():
            if p.requires_grad:
                count += p.numel()
        self._last_layer_name = 'classifier'
        self._last_dim = 4096
        self._last_dropout = 0.5
        self._last_activation = nn.ReLU(inplace=True)
        return count