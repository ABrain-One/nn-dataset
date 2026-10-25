import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU6(inplace=True)

        self.rep_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=out_channels, bias=False)

    def forward(self, x):
        if not self.deploy:
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        else:
            x = self.conv(x)

        x = self.rep_conv(x)
        return x

    def convert_to_deploy(self):
        if not self.deploy:
            self.rep_conv = nn.Conv2d(self.conv.out_channels, self.conv.out_channels, kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.out_channels, bias=False)
            self.conv = None
            self.bn = None
            self.relu = None
            self.deploy = True

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = [32, 64, 128, 256, 512]
        self.kernel_size = 3
        self.drop_path = 0.1
        self.device = device

        self.stem = RepVGGBlock(in_channels, self.channels[0], kernel_size=self.kernel_size, stride=2, padding=1)
        layers = []
        for i in range(1, len(self.channels)):
            layers.append(RepVGGBlock(self.channels[i-1], self.channels[i], kernel_size=self.kernel_size, stride=2, padding=1, deploy=False))

        self.layers = nn.Sequential(*layers)
        self.conv_head = nn.Conv2d(self.channels[-1], 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_head = nn.BatchNorm2d(256)
        self.relu = nn.ReLU6(inplace=True)
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.0))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def build_classifier(self, num_classes, dropout):
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.channels[-1], num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.conv_head(x)
        x = self.bn_head(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()