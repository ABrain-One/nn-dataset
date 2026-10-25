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
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )
        self._is_fwd_comp = False
        self._last_drops = []

    def build_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
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
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

    def auto_collect_statistics(self, inputs, prm):
        inputs = inputs.to(self.device)
        self.train()
        self._is_fwd_comp = True
        self._last_drops = []
        try:
            with torch.no_grad():
                _ = self(inputs)
        finally:
            self._is_fwd_comp = False
        return [(k, v.item()) for k, v in self.named_parameters() if k not in self._last_drops]

    def update_hyperparameters(self, prm):
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        return {'lr': lr, 'momentum': momentum}

    def get_loss_and_ppl(self, targets, inputs, prm):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.train()
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criteria[0](outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item(), None

    def get_optimizer(self):
        return self.optimizer