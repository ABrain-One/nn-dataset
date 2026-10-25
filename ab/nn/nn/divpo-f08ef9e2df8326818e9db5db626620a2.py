import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.build_layer(32, 32, 2, dropout=prm.get('dropout', 0.0))
        self.layer2 = self.build_layer(32, 64, 2, dropout=prm.get('dropout', 0.0))
        self.layer3 = self.build_layer(64, 128, 2, dropout=prm.get('dropout', 0.0))
        self.layer4 = self.build_layer(128, 256, 2, dropout=prm.get('dropout', 0.0))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.0)),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )
        self._is_fwd_computed = False
        self.fwd_shape = None
        self.compute_fwd_shape = in_shape is not None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self._last_prm = None
        self._last_input = None

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = self.build_optimizer(prm, self.parameters())

    def build_layer(self, in_channels, out_channels, blocks, kernel_size=3, padding=1, dropout=0.0):
        layers = []
        for i in range(blocks):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        if dropout > 0:
            layers += [nn.Dropout(p=dropout)]
        return nn.Sequential(*layers)

    def build_optimizer(self, prm, parameters):
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=5e-4)

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

    def forward(self, x):
        if not self._is_fwd_computed or self.compute_fwd_shape:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.flatten(1)
            self.fwd_shape = x.shape
            self._is_fwd_computed = True
        return self.classifier(x)

    def train_setup(self, prm):
        self._last_prm = prm
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = self.build_optimizer(prm, self.parameters())