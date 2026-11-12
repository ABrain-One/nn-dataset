import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        branch_features = out_channels
        hidden_features = in_channels * 4
        self.input_channels = in_channels

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, hidden_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_features),
                nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=2, padding=1, groups=hidden_features, bias=False),
                nn.BatchNorm2d(hidden_features),
                nn.Conv2d(hidden_features, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=stride, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.Conv2d(hidden_features, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        if self.stride == 1 and x.shape[2] == 28:
            x = self.relu(x)
            x_left = self.branch1(x)
            x_right = self.branch2(x)
            x = x_left + x_right
        else:
            x = self.branch1(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = InvertedResidual(32, 32, 2)
        self.block2 = InvertedResidual(32, 32, 2)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.2))
        self.momentum = prm.get('momentum', 0.9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=self.momentum,
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