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
                      kernel_size=(int(pi), 5), padding=0, bias=False),
            nn.BatchNorm2d(32),
        ]
        return nn.Sequential(*layers)

    def build_policy(self):
        layers = []
        for f in [32, 32]:
            layers += [
                nn.Conv2d(f, f, kernel_size=3, padding=1, groups=1),
                nn.Conv2d(f, f, kernel_size=(1, 7)),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
            ]
            layers += nn.Sequential(
                nn.Conv2d(f, f, kernel_size=1, bias=False),
                nn.BatchNorm2d(f)
            )
        self.policy = nn.Sequential(*layers)
        return self.policy

    def forward(self, x):
        x = self.features(x)
        x1 = self.policy(x)
        B, C, H, W = x1.size()
        x1 = nn.functional.adaptive_avg_pool2d(x1, (1, 1))
        return self.classifier(x1.squeeze(-1).squeeze(-1))

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for images, labels in train_data:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(images)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()