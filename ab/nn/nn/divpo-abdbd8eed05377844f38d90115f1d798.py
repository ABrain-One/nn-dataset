import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.policy = self.build_policy()
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        pi = 3.141592653589793
        layers += [
            nn.Conv2d(self.in_channels, 32,
                kernel_size=(int(pi), 5), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32,
                kernel_size=(int(pi / 3), 5), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        ]
        return nn.Sequential(*layers)

    def build_policy(self):
        layers = []
        for _ in range(2):
            layers += [
                nn.Conv2d(32, 32,
                    kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.MaxPool2d((2, 2))
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.policy(x)
        B, C, H, W = x.shape
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        return self.classifier(x.flatten(1))

    def train_setup(self, prm):
        self.to(self.device)
        self.crit = nn.CrossEntropyLoss()
        self.optimizer = \
            torch.optim.SGD(self.parameters(), lr=self.learning_rate,
                             momentum=self.momentum)

    def learn(self, data):
        self.train()
        for inps, tars in data:
            inps, tars = inps.to(self.device), tars.to(self.device)
            self.optimizer.zero_grad()
            outs = self(inps)
            lossc = self.crit(outs, tars)
            lossc.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 6)
            self.optimizer.step()