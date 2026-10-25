import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = int(out_shape[0])

        self.features = nn.Sequential(
            Block(self.in_channels, 64, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128, stride=2),
            Block(128, 256, stride=2),
            Block(256, 512, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.classifier = nn.Linear(512 * 6 * 6, self.num_classes)

        self._prepare_for_deployment(prm)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _prepare_for_deployment(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
(model := Net(in_shape=(3, 32, 32), out_shape=(10,), prm={'lr': 0.01, 'momentum': 0.9}, device=torch.device('cpu'))).to(torch.device('cpu'))
model.count_parameters() <= 500000