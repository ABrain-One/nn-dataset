import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def get_channels(in_shape):
    return in_shape[1]

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.channels = get_channels(in_shape)
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.conv1 = nn.Conv2d(self.channels, 64, kernel_size=3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.max_pool(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_device(self, device):
        self.to(device)

    def get_learning_rate(self):
        return self.learning_rate

    def get_momentum(self):
        return self.momentum