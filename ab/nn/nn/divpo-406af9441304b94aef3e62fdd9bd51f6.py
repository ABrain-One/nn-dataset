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
        self.layer1 = self.build_layer(32, 32, 3, 1, 1)
        self.layer2 = self.build_layer(32, 64, 3, 2, 1)
        self.layer3 = self.build_layer(64, 64, 3, 2, 1)
        self.layer4 = self.build_resblock(64, 64)
        self.layer5 = self.build_layer(64, 128, 3, 2, 1)
        self.layer6 = self.build_resblock(128, 128)
        self.layer7 = self.build_resblock(128, 128)
        self.layer8 = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.classifier = nn.Linear(512, out_shape[0])
        self._is_federated = False
        self._push_to_device(device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self._optimizer = None

    def build_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def build_resblock(self, in_channels, out_channels):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)] if in_channels != out_channels else []
        layers = nn.Sequential(*layers)
        return nn.Sequential(
            layers,
            nn.ReLU(inplace=True)
        )

    def _push_to_device(self, device):
        self.to(device)
        self.stem.to(device)
        for layer in self.layer1:
            layer.to(device)
        for layer in self.layer2:
            layer.to(device)
        for layer in self.layer3:
            layer.to(device)
        for layer in self.layer4:
            layer.to(device)
        for layer in self.layer5:
            layer.to(device)
        for layer in self.layer6:
            layer.to(device)
        for layer in self.layer7:
            layer.to(device)
        for layer in self.layer8:
            layer.to(device)
        self.classifier.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self._optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.02),
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
        )

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self._optimizer.zero_grad()
            output = self(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self._optimizer.step()

    def _is_federated(self):
        return self._is_federated