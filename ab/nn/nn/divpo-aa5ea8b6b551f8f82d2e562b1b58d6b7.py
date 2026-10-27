import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]  # number of target classes
        self.stem = nn.Sequential(
            Block(in_channels, 32, stride=2),
            Block(32, 64, stride=2),
            Block(64, 128, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.layers = nn.Sequential(
            Block(128, 128),
            Block(128, 128),
            Dropout(prm.get('dropout', 0.0)),
            Block(128, 128),
            Block(128, 128),
            Dropout(prm.get('dropout', 0.0)),
            Block(128, 128),
            Block(128, 128),
            Dropout(prm.get('dropout', 0.0))
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        def adopt_dropout(m):
            if type(m) is Block:
                m.dropout = prm.get('dropout', 0.0)

        self.layers.apply(adopt_dropout)

        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
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

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        return self.dropout(x)