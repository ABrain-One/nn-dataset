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
        self.layer1 = self.build_layer(32, 32, 2, first=True)
        self.layer2 = self.build_layer(32, 64, 2)
        self.layer3 = self.build_layer(64, 128, 2)
        self.layer4 = self.build_resblock(128, 128, 1)
        self.layer5 = self.build_resblock(128, 128, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._size = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.to(device)

        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizers(prm)

    def build_layer(self, in_planes, out_planes, stride, first=False):
        layers = []
        conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(out_planes)
        relu = nn.ReLU(inplace=True)
        layers += [conv1, bn1] if first else [conv1, bn1, relu]
        return nn.Sequential(*layers)

    def build_resblock(self, in_planes, out_planes, blocks):
        layers = []
        for i in range(blocks):
            layers += [
                self.build_layer(in_planes, out_planes, stride=1 if i == blocks - 1 else 2),
                self.build_layer(out_planes, out_planes, stride=1)
            ]
        return nn.Sequential(*layers)

    def configure_optimizers(self, prm):
        lr, momentum = prm['lr'], prm['momentum']
        return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)

    def train_setup(self, prm):
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = self.configure_optimizers(prm)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

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