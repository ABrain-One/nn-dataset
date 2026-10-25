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
        self._is_fwd_comp = False
        self._last_mu = None
        self.to(device)
        for p in self.parameters():
            nn.init.normal_(p, mean=0, std=0.01)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = self.build_optimizer(prm)

    
    def build_layer(self, in_channels, out_channels, blocks, **kw):
        layers = []
        for i in range(blocks):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def build_optimizer(self, prm):
        return torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 5e-4)
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
            if self._is_fwd_comp:
                self._last_mu = self.update_mu(self._last_mu, inputs, labels)

    def update_mu(self, mu, inputs, labels):
        raise NotImplementedError()

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
        self.optimizer = self.build_optimizer(prm)