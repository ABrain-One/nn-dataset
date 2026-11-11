import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InvertedResidual, self).__init__()
        if stride not in [1, 2]:
            raise ValueError("Invalid stride!")

        self.stride = stride

        branch_features = out_channels
        if in_channels == out_channels and stride == 1:
            branch_features = in_channels

        self.branch1 = nn.Sequential()
        if stride == 2:
            self.branch1.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_features, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        if branch1.shape == branch2.shape:
            x = branch1 + branch2
        else:
            x = torch.cat([branch1, branch2], dim=1)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.prm = prm
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        dropout_rate = prm.get('dropout', 0.2)
        self.channels = [32, 64, 128, 256]
        self.inverted_residual = InvertedResidual(32, 32, stride=1)
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            self.inverted_residual
        )
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.mean(dim=(2, 3))
        x = self.dropout(x)
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