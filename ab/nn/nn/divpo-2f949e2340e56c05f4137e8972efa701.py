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
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.build_layer(32, 32, 2, first=True)
        self.layer2 = self.build_layer(32, 64, 2)
        self.layer3 = self.build_layer(64, 128, 2)
        self.layer4 = self.build_resblock(128, 128, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_shape[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.criteria = (nn.CrossEntropyLoss().to(device),)
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
        )

    def build_layer(self, in_channels, out_channels, stride, first=False):
        layer = []
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(out_channels)

        if not first:
            layer += [conv1, bn1, relu]
        else:
            layer += [conv1, bn1, relu, nn.MaxPool2d(kernel_size=2, stride=2)]
        layer += [conv2, bn2, relu]
        return nn.Sequential(*layer)

    def build_resblock(self, in_channels, out_channels, repeat):
        layers = []
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True)]
        for _ in range(repeat - 1):
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return ['lr', 'momentum']