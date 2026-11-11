import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: str) -> None:
        super().__init__()
        self.device = device
        self.prms = prm
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.channels = 32
        self.depth = 28
        self.stem = nn.Conv2d(self.in_channels, self.channels, 3, padding=1)
        self.body = nn.ModuleList([Block(self.channels, self.channels) for _ in range(self.depth)])
        self.classifier = nn.Linear(self.channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(x)
        for block in self.body:
            features = block(features)
        logits = self.classifier(torch.mean(features, dim=(2, 3)))
        return logits

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()


class Block(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.mish = Mish()
        self.channels = channels
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.mish(x))


class Mish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (torch.tanh(nn.functional.softplus(x)))