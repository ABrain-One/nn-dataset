import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class DPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DPNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class DPN131(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, num_blocks=3, growth_rate=32):
        super(DPN131, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(DPNBlock(growth_rate, growth_rate))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(growth_rate, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        model_class = DPN131
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]
        self.model = model_class(self.channel_number, self.class_number, num_blocks=3, growth_rate=32)

        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        # Layerwise LR strategy: llr3_4grp_top2_full
        _llr_params = list(self.named_parameters())
        _llr_n = len(_llr_params)
        _llr_ratios = [0.25, 0.25, 0.25, 0.25]
        _llr_mults = [0.01, 0.01, 1, 1]
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
        self.optimizer = optim.SGD(_llr_groups, lr=self.learning_rate, momentum=self.momentum)

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
