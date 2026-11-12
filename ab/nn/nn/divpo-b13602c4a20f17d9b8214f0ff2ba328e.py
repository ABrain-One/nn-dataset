import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn_shortcut = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.functional.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        shortcut = self.shortcut(identity)
        if hasattr(self, 'bn_shortcut'):
            shortcut = self.bn_shortcut(shortcut)
        
        out += shortcut
        out = nn.functional.relu(out)
        
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = []
        in_channels = in_shape[1]
        # First convolutional layer
        self.features.append(nn.Conv2d(in_channels, 16, kernel_size=3, padding=1))
        self.features.append(nn.BatchNorm2d(16))
        self.features.append(nn.ReLU(inplace=True))
        
        # Stacked bottleneck blocks
        self.features.append(BottleneckBlock(16, 64, stride=1))
        self.features.append(BottleneckBlock(64, 64, stride=1))
        self.features.append(BottleneckBlock(64, 128, stride=2))
        self.features.append(BottleneckBlock(128, 128, stride=1))
        self.features.append(BottleneckBlock(128, 256, stride=2))
        self.features.append(BottleneckBlock(256, 256, stride=1))
        self.features.append(BottleneckBlock(256, 512, stride=2))
        self.features.append(BottleneckBlock(512, 512, stride=1))
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, int(out_shape[0]))
        
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

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