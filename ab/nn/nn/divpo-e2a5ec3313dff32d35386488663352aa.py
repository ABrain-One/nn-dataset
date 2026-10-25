import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = Block(64, 128, kernel_size=3, stride=1, padding=1)
        self.block3 = Block(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = Block(128, 256, kernel_size=3, stride=1, padding=1)
        self.block5 = Block(256, 256, kernel_size=3, stride=1, padding=1)
        self.block6 = Block(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block7 = Block(256, 512, kernel_size=3, stride=1, padding=1)
        self.block8 = Block(512, 512, kernel_size=3, stride=1, padding=1)
        self.block9 = Block(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block10 = Block(512, 512, kernel_size=3, stride=1, padding=1)
        self.block11 = Block(512, 512, kernel_size=3, stride=1, padding=1)
        self.block12 = Block(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool2(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.pool4(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.pool5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()