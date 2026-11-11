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
                kernel_size=(int(pi), 5), padding=(int(pi//2), 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32,
                kernel_size=(7), padding=(int(7//2))),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        ]
        return nn.Sequential(*layers)

    def build_policy(self):
        layers = []
        for _ in range(5):
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(32, 32,
                    kernel_size=3, padding=1)),
                nn.ReLU(True)
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.policy(x)
        B, C, H, W = x.shape
        X = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        return self.classifier(X.squeeze(-1).squeeze(-1))

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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()