import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block1 = self._make_block(32, 64, 2, True)
        self.block2 = self._make_block(64, 128, 2, True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self._last_channels = 128

    def _make_block(self, in_channels, out_channels, layers, downsample):
        layers_list = []
        for i in range(layers):
            stride = 2 if i == 0 and downsample else 1
            layers_list += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        lr = prm.get('lr', 1e-3)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

                                   

### Budget: 500000 parameters
### Constraint: edge-friendly (low latency)
### Resource: channels-first (N, C, H, W)
### Edge-friendly hyper-parameters: [1.0, 2.0, 1.5, 2.5, 1.25]
### Edge-friendly spatial packings: [1.25, 1.5, 1.75, 2.0, 2.25]
### Edge-friendly window sizes: [3, 5, 7, 9, 11]
### Edge-friendly window strides: [2, 3, 4, 5, 6]
### Edge-friendly block sizes: [3, 5, 7, 9, 11]
### Edge-friendly layer ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly depth ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly feature ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly skip ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly attention ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly attention scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios: [0.25, 0.5, 0.75, 1.0, 1.25]
### Edge-friendly multi-head scales: [3, 5, 7, 9, 11]
### Edge-friendly multi-head ratios