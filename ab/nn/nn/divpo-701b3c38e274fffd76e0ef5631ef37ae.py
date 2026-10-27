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
        self.layer1 = self.build_layer(32, 32, 2, prm)
        self.layer2 = self.build_layer(32, 64, 2, prm, first=False)
        self.layer3 = self.build_layer(64, 128, 2, prm, first=False)
        self.layer4 = self.build_resblock(128, 128, prm, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.0)),
            nn.Linear(512, out_shape[0])
        )
        self._last_layer_name = 'classifier'

    def build_layer(self, in_channels, out_channels, stride, prm, first=True):
        layer_class = nn.Conv2d if first else nn.Conv2d
        layer_norm = nn.BatchNorm2d if first else nn.BatchNorm2d
        layer_act = nn.ReLU(inplace=True) if not first else nn.Sequential()
        layer = layer_class(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        return nn.Sequential(layer, layer_norm(out_channels), layer_act)

    def build_resblock(self, in_channels, out_channels, prm, repeat):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(repeat - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()