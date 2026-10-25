import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Net class with helper Block (if not already in the given code)
class Block(nn.Module):
    # Block implementation from the previous response
    def __init__(self, in_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.layer(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.kernel_size = 3
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        layers = []
        layers += [Block(32), Block(32)]
        layers += [nn.Dropout(p=0.3)]
        self.layers1 = nn.Sequential(*layers)
        layers = []
        layers += [Block(32), Block(32)]
        layers += [nn.Dropout(p=0.3)]
        self.layers2 = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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