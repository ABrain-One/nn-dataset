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
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.0)),
            nn.Linear(128 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._set_forget_bias(self.layer1)
        self._set_forget_bias(self.layer2)
        self._set_forget_bias(self.layer3)
        self._set_conv_init(self.layer1)
        self._set_conv_init(self.layer2)
        self._set_conv_init(self.layer3)
        self._set_fc_init(self.classifier)

    def _make_layer(self, inplanes, planes, blocks, stride=1, dropout=0.0):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(blocks - 1):
            layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def _set_conv_init(self, m):
        for layer in m:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def _set_fc_init(self, m):
        for layer in m:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                nn.init.constant_(layer.bias, 0)

    def _set_forget_bias(self, m):
        for layer in m:
            if isinstance(layer, nn.LSTM):
                for param in layer.bias_ih.parameters():
                    nn.init.constant_(param, 1)
                for param in layer.bias_hh.parameters():
                    nn.init.constant_(param, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

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