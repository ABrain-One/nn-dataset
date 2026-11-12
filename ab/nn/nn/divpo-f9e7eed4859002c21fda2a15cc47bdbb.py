import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:
            in_channels = in_shape[1]  # channels-first (C, H, W) for CIFAR-10 RGB
        else:
            in_channels = int(in_shape[0])  # channels-last (H, W, C) for CIFAR-10 RGB
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.layer1 = Block(32, 32, prm['dropout'] if 'dropout' in prm else 0.0)
        self.layer2 = Block(32, 64, prm['dropout'] if 'dropout' in prm else 0.0)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm['dropout'] if 'dropout' in prm else 0.2),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.layer2(x)
        x = self.avgpool(x)
        return self.classifier(x)

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
            self.optimizer.step()


# Optional: helper blocks
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x


def check_shape(shape):
    assert isinstance(shape, (list, tuple))
    assert len(shape) in (3, 4)
    if len(shape) == 4:
        assert shape[1] in {1, 3}  # channels-last: [H, W, C] or [B, H, W, C]
    assert shape[0] in {1, 32, 64}  # CIFAR-10: 3, channels-first: [C, H, W]

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)