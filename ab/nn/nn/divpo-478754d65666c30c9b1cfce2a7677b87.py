import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Define a simple block for residual connection
class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SimpleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Main Network Class
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        
        # Derive input channels from in_shape
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        
        # First convolutional layer
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.block1 = nn.Sequential(
            SimpleBlock(64, 64),
            SimpleBlock(64, 64)
        )
        self.block2 = nn.Sequential(
            SimpleBlock(64, 128, stride=2),
            SimpleBlock(128, 128)
        )
        self.block3 = nn.Sequential(
            SimpleBlock(128, 256, stride=2),
            SimpleBlock(256, 256)
        )
        
        # Fully connected layer
        num_classes = int(out_shape[0])
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
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