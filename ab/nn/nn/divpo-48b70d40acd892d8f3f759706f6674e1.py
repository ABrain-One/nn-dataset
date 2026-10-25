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
        self.layer1 = self._make_layer(32, 32, 2, drop_out_prob=0.2)
        self.layer2 = self._make_layer(32, 64, 2, drop_out_prob=0.3)
        self.layer3 = self._make_layer(64, 128, 2, drop_out_prob=0.4)
        self.layer4 = self._make_layer(128, 256, 2, drop_out_prob=0.5)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )
        self._is_federated = None
        self._last_layer_name = None

    def _make_layer(self, in_channels, out_channels, blocks, drop_out_prob):
        layers = []
        for i in range(blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Dropout(drop_out_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.flatten(1)
        out = self.classifier(out)
        return out

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

    def _federated_phase(self, prm):
        self._is_federated = True
        self._last_layer_name = 'federated_last_layer'
        for n, _ in reversed(list(self._last_layer_name)):
            if n not in prm:
                continue
            self._last_layer_name = n
            break
        if self._last_layer_name not in prm:
            return
        in_features = prm[self._last_layer_name]['in_features']
        self.classifier = nn.Linear(in_features, out_features=10)