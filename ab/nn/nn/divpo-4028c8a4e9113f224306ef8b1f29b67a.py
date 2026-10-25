import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # derived from in_shape: (batch, channels, height, width)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = self._make_layer(32, 64, 2, True)
        self.block2 = self._make_layer(64, 128, 2, True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def _make_layer(self, inplanes, planes, blocks, downsample):
        layers = [nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1 if downsample else 2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.MaxPool2d(kernel_size=2, stride=1)
        )]
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 1e-4)
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

# Expected size of output tensor: (batch, channels, height, width)
# Expected input size: (batch_size, 3, height, width)
# Expected input shape (channel-first): (channels=3, height, width)