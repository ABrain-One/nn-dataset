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
        self.layer1 = self._make_layer(32, 32, 2, dropout=prm.get('dropout', 0.0))
        self.layer2 = self._make_layer(32, 64, 2, dropout=prm.get('dropout', 0.0))
        self.layer3 = self._make_layer(64, 128, 2, dropout=prm.get('dropout', 0.0))
        self.layer4 = self._make_layer(128, 256, 2, dropout=prm.get('dropout', 0.0))
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.0)),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.0)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0]),
        )
        self._size_changed = True
        self._last_size = (6, 6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        self._size_changed = False

        self.to(device)

        self.criteria = (
            nn.CrossEntropyLoss().to(device),
        )
        self.optimizer = self.build_optimizer(
            prm, self.parameters()
        )

    def build_optimizer(self, prm, param_groups):
        return torch.optim.SGD(
            param_groups,
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def _make_layer(self, in_channels, out_channels, blocks, dropout):
        layers = []
        for i in range(blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def forward(self, x):
        x = self.stem(x)
        if self._size_changed:
            self._last_size = x.shape[2:]
            self._size_changed = False
        for layer in self.layer1:
            x = layer(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        for layer in self.layer2:
            x = layer(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        for layer in self.layer3:
            x = layer(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        for layer in self.layer4:
            x = layer(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )