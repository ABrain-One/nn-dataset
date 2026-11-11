import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_resampling = stride != 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            # dw
            nn.Conv2d(hidden_dim if expand_ratio != 1 else inp, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_resampling:
            x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, 1, 3),
            nn.AdaptiveAvgPool2d(6),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(32 * 6 * 6, out_shape[0])
        self.to(self.device)
        self._set_hyper(prm)

    def _set_hyper(self, prm):
        self.lr = prm['lr']
        self.momentum = prm['momentum']
        self.step = 0

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self._set_hyper(prm)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for idx, data in enumerate(train_data):
            if self.step % 30 == 0:
                x, y = data
                x.requires_grad_()
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                out = self(x)
                loss = self.criteria[0](out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
                self.optimizer.step()
            self.step += 1