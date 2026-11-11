import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Helper block for residual connections
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = []
        in_channels = in_shape[1]
        out_channels = 64
        
        # First convolution layer
        self.features.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1))
        self.features.append(nn.BatchNorm2d(64))
        self.features.append(nn.ReLU(inplace=True))
        
        # Two residual blocks
        self.features.append(ResidualBlock(64, 64))
        self.features.append(ResidualBlock(64, 64))
        
        # Third convolution layer to reduce spatial dimensions
        self.features.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.features.append(nn.BatchNorm2d(128))
        self.features.append(nn.ReLU(inplace=True))
        
        # Two more residual blocks
        self.features.append(ResidualBlock(128, 128))
        self.features.append(ResidualBlock(128, 128))
        
        # Final convolution layer to further reduce spatial dimensions
        self.features.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        self.features.append(nn.BatchNorm2d(256))
        self.features.append(nn.ReLU(inplace=True))
        
        # Two more residual blocks
        self.features.append(ResidualBlock(256, 256))
        self.features.append(ResidualBlock(256, 256))
        
        # Adaptive average pooling and dropout (optional)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        p = prm.get('dropout', 0.3)
        self.dropout = nn.Dropout(p=p) if p > 0 else nn.Identity()
        
        # Fully connected layer
        num_classes = int(out_shape[0])
        self.fc = nn.Linear(256 * 7 * 7, num_classes)
        
        self.features = nn.Sequential(*self.features)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
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