import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def learn_rate(prm):
    return prm['lr'] if 'lr' in prm else 0.01

def momentum(prm):
    return prm['momentum'] if 'momentum' in prm else 0.9

def create_block(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False)
        self.block1 = create_block(32)
        self.block2 = create_block(32)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, out_shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=learn_rate(prm),
            momentum=momentum(prm),
            weight_decay=0.0001
        )

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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_learning_rate(self):
        return learn_rate(self.prm)

    def get_momentum(self):
        return momentum(self.prm)