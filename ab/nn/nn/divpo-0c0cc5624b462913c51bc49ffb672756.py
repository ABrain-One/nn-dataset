import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SmallBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.stride = stride
        self.shortcut = nn.Sequential();
        if stride > 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            );
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        self.first_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.first_bn = nn.BatchNorm2d(32)
        self.first_relu = nn.ReLU(inplace=True)
        self.first_maxpool = nn.MaxPool2d(kernel_size=1, stride  =1)
        self.small1 = SmallBlock(32, 32, stride=1  )        
        self.small2 = SmallBlock(32, 34, stride=1  )        
        self.avgpool = nn.AvgPool2d(30)  # DERIVED: max of 28,30,32 for yx dimension after small2
        self.classifier = nn.Linear(34, out_shape[0])

    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_maxpool(x)
        x = self.small1(x)
        x = self.small2(x)
        x = self.avgpool(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()