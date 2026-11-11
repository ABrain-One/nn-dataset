import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple[int, int, int, int], out_shape: tuple[int], prm: dict[str, float], device: torch.device) -> None:
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = int(out_shape[0])
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Convolutional blocks
        self.block1 = self._make_block(32, 32, 2)
        self.block2 = self._make_block(32, 64, 2)
        self.block3 = self._make_block(64, 64, 2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)
        
    def _make_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        layers = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            layers.add_module('conv1x1', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))
            layers.add_module('bn1', nn.BatchNorm2d(out_channels))
        layers.add_module('conv3x3', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.add_module('bn2', nn.BatchNorm2d(out_channels))
        layers.add_module('relu', nn.ReLU(inplace=True))
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def train_setup(self, prm: dict[str, float]) -> None:
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )
    
    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()