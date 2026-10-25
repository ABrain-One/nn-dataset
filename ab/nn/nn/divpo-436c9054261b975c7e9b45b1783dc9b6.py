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
        self.layer4 = self._make_layer(128, 256, 2, dropout=prm.get('dropout', 0.0))
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])
        self._last_channel = 256 * 6 * 6
        self._last_stride = 2
        self._last_dilation = 1
        self._last_kernel_size = 1
        self._last_padding = 0
        self._last_out_channels = 256
        self._last_groups = 1
        self._last_transposed = False
        self._last_output_padding = 0
        self._last_ceil_mode = False
        self._last_divisible_by = 1
        self._last_norm_layer = None
        self._last_act_layer = nn.ReLU(inplace=True)
        self._last_norm_kwargs = {}
        self._last_act_kwargs = {}

    def _make_layer(self, inplanes, planes, blocks, stride=1, dropout=0.0):
        layers = []
        layers += [
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers += [nn.Dropout(p=dropout)]
        for _ in range(blocks - 1):
            layers += [
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
        layers += [
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
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
            self.optimizer.step()