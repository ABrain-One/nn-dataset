import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = Block(32, 32, kernel_size=3, padding=1)
        self.layer2 = Block(32, 64, kernel_size=3, stride=2, padding=1)

        self.layers3 = nn.Sequential(
            Block(64, 64, kernel_size=3, padding=1),
            Block(64, 64, kernel_size=3, padding=1)
        )

        self.layers4 = nn.Sequential(
            Block(64, 128, kernel_size=3, stride=2, padding=1),
            Block(128, 128, kernel_size=3, padding=1)
        )

        self.layers5 = nn.Sequential(
            Block(128, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.layers5(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

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

# Dummy input to infer shape and check total params
in_shape = (3, 32, 32)
prm = {'lr': 0.01, 'momentum': 0.9}
device = torch.device('cpu')
dummy_input = torch.randn(1, *in_shape)
model = Net(in_shape, out_shape=(10,), prm=prm, device=device)
model.train_setup(prm)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total_params}')