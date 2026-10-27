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

    def fuse(self):
        # Fused weights for edge-friendly deployment
        if self.conv.weight.shape[1] == 3:
            fused_weight = torch.empty(3, *self.conv.weight.shape[1:], device=self.conv.weight.device)
            fused_weight[0] = self.conv.weight[0]
            fused_weight[1] = self.conv.weight[1]
            fused_weight[2] = self.conv.weight[2]
            self.conv.weight = nn.Parameter(fused_weight)
        self.bn.weight = nn.Parameter(self.bn.running_mean)
        self.bn.bias = nn.Parameter(self.bn.running_bias)
        self.bn.num_batches_tracked = nn.Parameter(torch.tensor(0, device=self.bn.running_mean.device))
        self.__class__ = nn.Conv2d

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Derive input channels from in_shape (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.layer1 = Block(32, 64, stride=1)
        self.layer2 = nn.Sequential(
            Block(64, 64, stride=2),
            Block(64, 64, stride=1)
        )
        self.layer3 = Block(64, 128, stride=2)
        self.layer4 = nn.Sequential(
            Block(128, 128, stride=2),
            Block(128, 128, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
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