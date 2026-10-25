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
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build_layer(128, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._is_fwd_comp = False
        self._last_inner = None

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

    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        for i in range(blocks):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride if i == 0 else 1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def compile(self, prm):
        self._is_fwd_comp = True
        self._last_inner = None
        self._last_inner_shape = None
        self._last_inner_out = None
        self._last_inner_is_linear = False
        self._last_inner_weight = None
        self._last_inner_bias = None

        if 'kernel_size' in prm:
            self._last_inner = nn.Conv2d(
                in_channels=self._last_inner_out,
                out_channels=prm['out_channels'],
                kernel_size=prm['kernel_size'],
                stride=prm.get('stride', 1),
                padding=prm.get('padding', 0)
            )
            self._last_inner_out = prm['out_channels']
            self._last_inner_shape = [prm['out_channels'], prm['kernel_size'], prm['kernel_size']]
            self._last_inner_is_linear = False
        else:
            self._last_inner = nn.Linear(
                in_features=self._last_inner_out,
                out_features=prm['out_features']
            )
            self._last_inner_out = prm['out_features']
            self._last_inner_shape = [prm['out_features']]
            self._last_inner_is_linear = True

        self._last_inner.weight.data.copy_(self._last_inner_weight)
        if self._last_inner_bias is not None:
            self._last_inner.bias.data.copy_(self._last_inner_bias)

        self.to_deploy()

    def to_deploy(self):
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = self._last_inner
        self.features = nn.Sequential(*list(self.children())[:-1])
        self.classifier = nn.Identity()