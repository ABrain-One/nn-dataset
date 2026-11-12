import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DoubleConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        
        self.conv1 = DoubleConvBlock(self.in_channels, 64, stride=1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConvBlock(64, 128, stride=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = DoubleConvBlock(128, 256, stride=1)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = DoubleConvBlock(256, 512, stride=1)
        self.drop_out = nn.Dropout(prm.get('dropout', 0.5)) if 'dropout' in prm else nn.Identity()
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 7 * 7, self.num_classes)
        
        self.train_setup(prm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.drop_out(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0)
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