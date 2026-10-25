import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(in_channels, out_channels, stride=1, groups=1, eps=1e-5, momentum=0.1, affine=True, is_training=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine),
        nn.ReLU(inplace=True) if is_training else nn.ReLU6(inplace=True)
    )

def supported_hyperparameters():
    # Only list the hyperparameters you really use in the code
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv_bn(in_channels, out_channels, stride=1, groups=1)
        self.conv2 = conv_bn(out_channels, out_channels, stride=1, groups=1)
        self.conv3 = conv_bn(out_channels, out_channels, stride=1, groups=1)
        self.conv4 = conv_bn(out_channels, out_channels, stride=1, groups=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1)
        )
        self.layer1 = nn.Sequential(
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1)
        )
        self.layer2 = nn.Sequential(
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1)
        )
        self.layer3 = nn.Sequential(
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1)
        )
        self.layer4 = nn.Sequential(
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1),
            conv_bn(32, 32, stride=1, groups=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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