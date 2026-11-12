import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.conv_1 = conv_bn(in_channels, in_channels * expand_ratio, kernel_size=1, stride=1, padding=0)
        self.conv_2 = conv_bn(in_channels * expand_ratio, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        if self.dropout_rate > 0:
            x = nn.functional.dropout2d(x, p=self.dropout_rate)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.dropout_rate = prm.get('dropout', 0.3)
        self.in_channels = in_channels
        self.input_size = 32
        self.block = Block(in_channels, in_channels, 2, 6, self.dropout_rate)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)
        self.device = device

    def forward(self, x):
        x = self.block(x)
        x = self.avg_pool(x)
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