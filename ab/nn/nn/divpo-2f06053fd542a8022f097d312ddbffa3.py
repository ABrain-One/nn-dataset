import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = int(out_shape[0])
        
        self.first_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.first_conv.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.block1 = DoubleConvBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.block2 = DoubleConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.block3 = DoubleConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 1024, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        self.bn2 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.first_conv(x)))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.bn2(self.fc1(x)))
        x = self.fc2(x)
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