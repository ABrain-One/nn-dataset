import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # Deriving input channels
        num_classes = int(out_shape[0])  # Number of target classes
        
        # First convolution layer
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.stem.weight, mode='fan_out', nonlinearity='relu')
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.block1 = self._make_block(32, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = self._make_block(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = self._make_block(64, 128, 5)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Fully connected layer
        self.fc = nn.Linear(128 * 6 * 6, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(prm.get('dropout', 0.3))

    def _make_block(self, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(Block(in_channels, out_channels))
            else:
                layers.append(Block(out_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.relu(self.bn1(x))
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = nn.functional.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(identity)
        x = self.relu(x)
        return x