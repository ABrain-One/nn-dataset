import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, expansion=4):
        super().__init__()
        hidden_dim = int(out_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.layer1 = Block(32, 32, 3, 1, 1, 1)
        self.layer2 = Block(32, 32, 3, 1, 1, 1)
        self.layer3 = Block(32, 64, 3, 2, 1, 1)
        self.layer4 = Block(64, 64, 3, 1, 1, 1)
        self.layer5 = Block(64, 64, 3, 1, 1, 1)
        self.layer6 = Block(64, 96, 3, 1, 1, 2)
        self.layer7 = Block(96, 96, 3, 1, 1, 1)
        self.layer8 = Block(96, 96, 3, 1, 1, 1)
        self.layer9 = Block(96, 192, 3, 2, 1, 4)
        self.layer10 = Block(192, 192, 3, 1, 1, 1)
        self.layer11 = Block(192, 192, 3, 1, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()