import torch
import torch.nn as nn


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        # Layerwise LR strategy: llr2_4grp_cos
        _llr_params = list(self.named_parameters())
        _llr_n = len(_llr_params)
        _llr_ratios = [0.25, 0.25, 0.25, 0.25]
        _llr_mults = [0.1, 0.325, 0.775, 1]
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
        self.optimizer = torch.optim.SGD(_llr_groups, lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        layers = []
        in_channels = in_shape[1]
        layers += [
            nn.Conv2d(in_channels, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
        ]
        in_channels = 384
        layers += [
            nn.Conv2d(in_channels, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
        ]
        in_channels = 384
        layers += [
            nn.Conv2d(in_channels, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ELU(inplace=True),
        ]
        layers.append(nn.AdaptiveMaxPool2d(output_size=(6, 6)))
        in_channels = 192
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm['dropout']),
            nn.Linear(classifier_input_features, 3072),
            nn.ELU(inplace=True),
            nn.Dropout(p=prm['dropout']),
            nn.Linear(3072, 2048),
            nn.ELU(inplace=True),
            nn.Linear(2048, out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Chromosome used to generate this model:
# {'conv1_filters': 96, 'conv1_kernel': 11, 'conv1_stride': 4, 'conv2_filters': 256, 'conv2_kernel': 5, 'conv3_filters': 384, 'conv4_filters': 384, 'conv5_filters': 192, 'fc1_neurons': 3072, 'fc2_neurons': 2048, 'lr': 0.01, 'momentum': 0.95, 'dropout': 0.6, 'include_conv1': 0, 'include_conv2': 0, 'include_conv3': 1, 'include_conv4': 1, 'include_conv5': 1, 'pooling_type1': 'AdaptiveMaxPool2d', 'pooling_type2': 'MaxPool2d', 'pooling_type3': 'AdaptiveMaxPool2d', 'activation_type': 'ELU', 'use_batchnorm': 1}
