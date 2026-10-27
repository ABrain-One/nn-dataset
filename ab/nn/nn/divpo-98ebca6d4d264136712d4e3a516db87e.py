import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(32)
        self.layer1 = self.build_layer(32, 32, 3, 1, 1)
        self.layer2 = self.build_layer(32, 64, 3, 2, 1)
        self.layer3 = self.build_layer(64, 64, 3, 2, 1)
        self.layer4 = self.build_resblock(64, 64, 3, 1, 1)
        self.layer5 = self.build_resblock(64, 64, 3, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, out_shape[0])
        for _ in self.modules():
            if isinstance(_, nn.Conv2d):
                torch.nn.init.kaiming_normal_(_.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(_, nn.BatchNorm2d):
                _.weight.data.fill_(1)
                _.bias.data.zero_()
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert params <= 500000, "Total number of parameters must be less than or equal to 500000."
        self.to(device)
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                assert isinstance(module.weight, torch.Tensor), f"Weight is not a Tensor: {module}"
                assert module.weight.requires_grad, "Weight does not require grad."
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def build_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def build_resblock(self, in_channels, out_channels, kernel_size, stride, padding):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        self.skip = stride == 1 and in_channels == out_channels
        if self.skip:
            layers.append(nn.Identity())
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.batchnorm(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

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