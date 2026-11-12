import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = int(out_shape[0])
        
        self.first_conv = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1)
        self.b1 = DoubleConvBlock(64, 64, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.b2 = DoubleConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.b3 = DoubleConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.b4 = DoubleConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.b5 = DoubleConvBlock(512, 512)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(512 * 7 * 7, self.num_classes)
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.b1(x)
        x = self.pool1(x)
        x = self.b2(x)
        x = self.pool2(x)
        x = self.b3(x)
        x = self.pool3(x)
        x = self.b4(x)
        x = self.pool4(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

# Example usage:
# model = Net((3, 32, 32), (10,), {'lr': 0.001, 'momentum': 0.9}, 'cpu')
# model.train_setup({'lr': 0.001, 'momentum': 0.9})
# model.learn(train_loader)