import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, expansion=1):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels)
            ) if in_channels != out_channels or stride != 1 else None)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.act(x + residual)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1)
        self.block1 = Block(32, 32, kernel_size=3, padding=1, stride=1, expansion=1)
        self.block2 = Block(32, 32, kernel_size=3, padding=1, stride=1, expansion=1)
        self.block3 = Block(32, 64, kernel_size=3, padding=1, stride=2, expansion=1)
        self.block4 = Block(64, 64, kernel_size=3, padding=1, stride=1, expansion=1)
        self.block5 = Block(64, 64, kernel_size=3, padding=1, stride=1, expansion=1)
        self.block6 = Block(64, 96, kernel_size=3, padding=1, stride=2, expansion=1)
        self.block7 = Block(96, 96, kernel_size=3, padding=1, stride=1, expansion=1)
        self.block8 = Block(96, 96, kernel_size=3, padding=1, stride=1, expansion=1)
        self.block9 = Block(96, 96, kernel_size=3, padding=1, stride=1, expansion=1)
        self.block10 = Block(96, 128, kernel_size=3, padding=1, stride=2, expansion=1)
        self.block11 = Block(128, 128, kernel_size=3, padding=1, stride=1, expansion=1)
        self.block12 = Block(128, 128, kernel_size=3, padding=1, stride=1, expansion=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.avgpool(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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