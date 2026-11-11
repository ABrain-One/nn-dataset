import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.layer.forward(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(6),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 500)
        )
        self.classifier = nn.Linear(500, out_shape[0])
        self.to(device); self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'] if 'lr' in prm else 0.01, momentum=prm['momentum'] if 'momentum' in prm else 0.9)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device); self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'] if 'lr' in prm else 0.01, momentum=prm['momentum'] if 'momentum' in prm else 0.9)

    def learn(self, train_data):
        self.train(); self.optimizer.zero_grad()
        for times, (x, y) in enumerate(train_data):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            outputs = self(x)
            loss = self.criteria[0](outputs, y)
            if times % 10 == 0: loss.backward(torch.ones_like(loss) * 20); nn.utils.clip_grad_norm_(self.parameters(), 3); self.optimizer.step()
        self.optimizer.step()