import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1)
        self.layer1 = self.build_layer(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = self.build_layer(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer3 = self.build_layer(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer4 = self.build_layer(128, 128, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self._set_property_from_prm(prm, 'lr', 'momentum', 'weight_decay')

    def build_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

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

    def _set_property_from_prm(self, prm, key, *keys):
        for k in keys:
            if k in prm:
                setattr(self, k, prm[k])