import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels // 8),
            nn.ELU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.layer.forward(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        channels = in_shape[1]
        self.device = device
        self.project = nn.Sequential(
            nn.Conv2d(channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        self.block = Block(32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.classify = nn.Linear(32, out_shape[0])

    def forward(self, x):
        # (500000 <= 412883 * 1.024)
        x = self.project(x)
        x = self.block(x)
        x = self.gap(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.055124733268068706) / 3.33
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=prm.get('momentum', 0.8483911545316851)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criteria[0](self.forward(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3); self.optimizer.step()