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
        self.layer1 = self._make_layer(32, 32, 2, drop_out_prob=0.2)
        self.layer2 = self._make_layer(32, 64, 2, drop_out_prob=0.3)
        self.layer3 = self._make_layer(64, 128, 2, drop_out_prob=0.4)
        self.layer4 = self._make_layer(128, 256, 2, drop_out_prob=0.5)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )
        self._is_fwd_complied = False
        self._last_prm = None

    def _make_layer(self, in_channels, out_channels, blocks, drop_out_prob):
        layers = []
        for i in range(blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.Dropout(drop_out_prob))
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
        self._last_prm = prm
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def _compile(self):
        if self._is_fwd_complied: return
        self._is_fwd_complied = True
        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = self
            def forward(self, x):
                return self.net(x)
        self = _Wrapper()
        self.cuda(device=0)
        inp = torch.randn((1, *self.stem.conv1.weight.shape[1:]), device='cuda:0')
        self(inp).sum().backward()