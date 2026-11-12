import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Involution(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=kernel_size // 2, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.involution = Involution(in_channels, kernel_size=3)
        self.act = nn.ReLU(inplace=True)
        self.conv_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.involution(x)
        out = self.act(out)
        out = self.conv_proj(out)
        out = out + identity
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.classifier = nn.Linear(500, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = None

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3);
            self.optimizer.step()