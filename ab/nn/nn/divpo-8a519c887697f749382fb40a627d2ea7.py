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
        self.layer1 = self._make_layer(32, 32, 2, 1)
        self.layer2 = self._make_layer(32, 64, 2, 2)
        self.layer3 = self._make_layer(64, 128, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )
        self._last_linear_channels(self.classifier)
        self._last_linear_units(self.classifier)
        self._last_linear(self.classifier)
        self._last_relu(self.classifier)
        self._last_batchnorm(self.classifier)
        self._last_conv(self.classifier)
        self._last_pool(self.classifier)
        self._last_relu(self.stem)
        self._last_batchnorm(self.stem)
        self._last_conv(self.stem)
        self._last_pool(self.stem)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = self._choose_optimizer(prm, self.parameters())

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        for _ in range(blocks - 1):
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def _choose_optimizer(self, prm, parameters):
        return torch.optim.SGD(parameters, lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def _last_linear_channels(self, module):
        if isinstance(module[-1], nn.Linear):
            self.in_channels = module[-1].weight.shape[1]

    def _last_linear_units(self, module):
        if isinstance(module[-1], nn.Linear):
            self.num_classes = module[-1].weight.shape[0]

    def _last_linear(self, module):
        if isinstance(module[-1], nn.Linear):
            self.last_linear = module[-1]

    def _last_relu(self, module):
        if isinstance(module[-1], nn.ReLU):
            self.last_relu = module[-1]

    def _last_batchnorm(self, module):
        if isinstance(module[-1], nn.BatchNorm2d):
            self.last_batchnorm = module[-1]

    def _last_conv(self, module):
        if isinstance(module[-1], nn.Conv2d):
            self.last_conv = module[-1]

    def _last_pool(self, module):
        if isinstance(module[-1], nn.MaxPool2d):
            self.last_pool = module[-1]

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}