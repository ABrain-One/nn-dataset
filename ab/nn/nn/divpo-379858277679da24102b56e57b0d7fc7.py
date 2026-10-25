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
        self.layer1 = self.build_layer(32, 32, 2, stride=1)
        self.layer2 = self.build_layer(32, 64, 2, stride=2)
        self.layer3 = self.build_layer(64, 128, 2, stride=2)
        self.layer4 = self.build_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._is_fwd_computed = False
        self.fwd_out_channels = 256

        self._count_params()

    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(blocks - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        if not self._is_fwd_computed:
            self._compute_fwd_out_channels(x)
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
            self.optimizer.step()

    def _compute_fwd_out_channels(self, x):
        with torch.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            self.fwd_out_channels = x.shape[1]

    def _count_params(self):
        self._params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_name_postfix(self):
        return f"v{int(self._params / 100000)}l{len(self.stem) + len(self.layer1) + len(self.layer2) + len(self.layer3) + len(self.layer4)}"

    def __str__(self):
        return f"ConvNet({self._get_name_postfix()})"

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}