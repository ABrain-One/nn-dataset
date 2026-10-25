import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def get_3x3_conv(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, bias: bool = False) -> nn.Conv2d:
    """Helper function to get a 3x3 convolution layer with the provided parameters"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias)

def get_5x5_conv(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, bias: bool = False) -> nn.Conv2d:
    """Helper function to get a 5x5 convolution layer with the provided parameters"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, groups=groups, bias=bias)

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # channels-first (C, H, W)
        self.num_classes = out_shape[0]
        
        self.conv1 = get_3x3_conv(self.in_channels, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer1 = self.make_layer(32, 32, 3, 1)
        self.layer2 = self.make_layer(32, 64, 3, 2)
        self.layer3 = self.make_layer(64, 128, 3, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)
        
    def make_layer(self, in_channels: int, out_channels: int, block_num: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(get_3x3_conv(in_channels, out_channels, stride))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(block_num - 1):
            layers.append(get_3x3_conv(out_channels, out_channels, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )
    
    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model(self):
        return self

    def get_input_shape(self):
        return (3, 32, 32)

    def supported_hyperparameters(self):
        return ['lr', 'momentum']