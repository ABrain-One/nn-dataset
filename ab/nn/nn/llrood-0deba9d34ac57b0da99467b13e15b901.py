import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        # Layerwise LR strategy: llr3_2grp_05x
        _llr_params = list(self.named_parameters())
        _llr_n = len(_llr_params)
        _llr_ratios = [0.5, 0.5]
        _llr_mults = [0.5, 1]
        _llr_groups = []
        _llr_start = 0
        for _llr_i, (_llr_r, _llr_m) in enumerate(zip(_llr_ratios, _llr_mults)):
            if _llr_i < len(_llr_ratios) - 1:
                _llr_size = max(1, round(_llr_n * _llr_r))
            else:
                _llr_size = _llr_n - _llr_start
            _llr_end = min(_llr_start + _llr_size, _llr_n)
            if _llr_start < _llr_n:
                _llr_groups.append({'params': [p for _, p in _llr_params[_llr_start:_llr_end]], 'lr': prm.get('lr', 0.01) * _llr_m})
            _llr_start = _llr_end
        self.optimizer = torch.optim.SGD(_llr_groups, lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def _make_layer(self, inplanes, planes, blocks):
        layers = []
        layers.append(Block(inplanes, planes))
        for _ in range(1, blocks):
            layers.append(Block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
            self.optimizer.step()

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
