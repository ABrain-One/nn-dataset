import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
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

# This code is part of a template for PyTorch model definitions.
# The target is to create a neural network that achieves the best accuracy after the first epoch of training.
# The produced model must be entirely defined within this scope.
# You are required to analyze the input and output shapes of the provided blocks and decide on the architecture.
# The goal is to achieve the best accuracy in the first epoch of training.
# You may use standard PyTorch layers and functions. Standard layer and module types include:
# nn.BatchNorm2d, nn.ReLU, nn.Conv2d, nn.Linear, nn.Dropout, and those available in the provided blocks.
# You may not use any of the following: nn.CrossEntropyLoss, F.cross_entropy, torch.nn.functional.interpolate, F.interpolate, F.adaptive_avg_pool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.Upsample, F.max_pool1d, F.max_pool2d, F.max_pool3d, F.avg_pool1d, F.avg_pool2d, F.avg_pool3d, F.avg_pool2d, F.avg_pool3d, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate, F.interpolate