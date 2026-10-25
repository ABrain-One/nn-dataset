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
        self._set_forget_bias(self.classifier[1])
        self._set_weight_decay(self.classifier[1], prm.get('weight_decay', 0.0))
        self._set_dropout(self.classifier[0], prm.get('dropout', 0.0))
        self._set_weight_decay(self.classifier[0], prm.get('weight_decay', 0.0))
        self._set_weight_decay(self, prm.get('weight_decay', 0.0))
        self._set_dropout(self, prm.get('dropout', 0.0))
        self._set_bias(self.stem[-1], False)
        self._set_bias(self.stem[1], False)
        self._set_bias(self.layer1[-1], False)
        self._set_bias(self.layer2[-1], False)
        self._set_bias(self.layer3[-1], False)
        self._set_bias(self.classifier[-1], False)
        self._set_bias(self.stem[0], False)
        self._set_bias(self.layer1[0], False)
        self._set_bias(self.layer2[0], False)
        self._set_bias(self.layer3[0], False)
        self._set_bias(self.classifier[1], False)
        self._set_bias(self.classifier[0], False)
        self._set_bias(self.avg_pool, False)
        self._set_bias(self, False)
        self._set_weight_decay(self, prm.get('weight_decay', 0.0))
        self._set_weight_decay(self.stem[0], prm.get('weight_decay', 0.0))
        self._set_weight_decay(self.stem[1], prm.get('weight_decay', 0.0))
        self._set_weight_decay(self.layer1[0], prm.get('weight_decay', 0.0))
        self._set_weight_decay(self.layer2[0], prm.get('weight_decay', 0.0))
        self._set_weight_decay(self.layer3[0], prm.get('weight_decay', 0.0))
        self._set_weight_decay(self.classifier[1], prm.get('weight_decay', 0.0))
        self._set_weight_decay(self.classifier[0], prm.get('weight_decay', 0.0))
        self._set_weight_decay(self.avg_pool, prm.get('weight_decay', 0.0))

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

        self._to_v2(self.stem)
        self._to_v2(self.layer1)
        self._to_v2(self.layer2)
        self._to_v2(self.layer3)
        self._to_v2(self.classifier)

        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
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
        layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def _to_v2(self, m):
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            m._version = 2

    def _set_weight_decay(self, m, weight_decay):
        if hasattr(m, 'weight') and not hasattr(m, 'bias'):
            nn.utils.weight_norm(m, name='weight', dim=0)
            m._weight_decay = weight_decay

    def _set_dropout(self, m, dropout):
        if isinstance(m, nn.Dropout):
            m.p = dropout

    def _set_bias(self, m, bias):
        if hasattr(m, 'bias') and bias is False:
            m.bias.requires_grad = False

    def _set_forget_bias(self, m):
        if isinstance(m, nn.RNNBase):
            m.bias_ih.data.fill_(1)

    def _set_forget_bias(self, m):
        if isinstance(m, nn.RNNBase):
            m.bias_ih.data.fill_(1)