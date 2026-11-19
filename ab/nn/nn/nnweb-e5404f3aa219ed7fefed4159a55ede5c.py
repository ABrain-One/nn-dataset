import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        inputs: int = in_shape[1]
        outputs: int = out_shape[0]
        self.num_classes = outputs

        # ResNet-9 configuration
        # Width scale 0.75 -> base channels 32 * 0.75 = 24
        width_scale = 0.75
        base_channels = int(32 * width_scale)

        self.conv1 = nn.Conv2d(inputs, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # ResNet-9 has 3 stages of 1 block each (1+1+1 = 3 blocks)
        # Stage 1: base_channels -> base_channels, stride 1
        self.layer1 = BasicBlock(base_channels, base_channels, stride=1)
        
        # Stage 2: base_channels -> 2*base_channels, stride 2
        self.layer2 = BasicBlock(base_channels, base_channels * 2, stride=2)
        
        # Stage 3: 2*base_channels -> 4*base_channels, stride 2
        self.layer3 = BasicBlock(base_channels * 2, base_channels * 4, stride=2)

        self.drop = nn.Dropout(prm.get('dropout', 0.0))
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(base_channels * 4, outputs)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.drop(out)
        out = self.fc(out)
        return out


def supported_hyperparameters():
    return {'lr', 'momentum'}