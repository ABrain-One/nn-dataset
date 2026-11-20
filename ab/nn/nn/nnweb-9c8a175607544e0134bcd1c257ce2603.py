import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'dropout_aux'}

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes = out_shape[0]
        dropout = prm['dropout']
        dropout_aux = prm['dropout_aux']

        # Standard ResNet-9 for CIFAR-10 (32x32 input)
        # Structure: Conv-BN-ReLU -> 3 Residual Blocks (32x32) -> AvgPool -> FC
        # Depth 9: 1 initial conv + 3 blocks * 2 convs each + 1 final fc = 9 layers
        
        # Initial Conv
        # Standard CIFAR ResNet uses 64 channels for the first conv
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual Blocks
        # Standard ResNet-9 typically has 3 blocks of 2 layers each.
        # No downsampling in these blocks to keep 32x32 spatial size.
        self.block1 = ResidualBlock(64, 64, stride=1)
        self.block2 = ResidualBlock(64, 64, stride=1)
        self.block3 = ResidualBlock(64, 64, stride=1)

        # Head
        # Standard CIFAR ResNet uses Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout is required by hyperparameters. 
        # Standard ResNet-9 often doesn't have dropout, but we must use it.
        # We place it before the final FC.
        self.dropout = nn.Dropout(p=dropout)
        
        self.fc = nn.Linear(64, num_classes)
        
        # We must use dropout_aux. In ResNet-9, there are no aux heads.
        # To satisfy the requirement "use every hyperparameter", we will apply 
        # dropout_aux to the features before the final FC as well, or simply 
        # use it in a dummy way? No, "actually used".
        # Let's apply dropout_aux to the output of the blocks before pooling? 
        # Or better, since ResNet-9 is simple, we can just use it for the main dropout 
        # if the other isn't used? No, they are distinct.
        # Let's use dropout_aux on the features before the final FC, and dropout on the final logits? 
        # Or just combine them? 
        # The prompt says: "each of those names ... must still be read from prm and actually used".
        # I will apply `dropout` to the features after pooling, and `dropout_aux` to the features before pooling? 
        # That seems arbitrary. 
        # Alternatively, standard ResNet-9 implementations often don't have aux. 
        # I will use `dropout` for the main path dropout. 
        # I will use `dropout_aux` to drop out the features before the final FC as well? 
        # Let's just use `dropout_aux` for the dropout layer before FC, and `dropout` for... wait.
        # Let's use `dropout` for the layer before FC. 
        # What about `dropout_aux`? 
        # I will add a second dropout layer with p=dropout_aux on the same features? 
        # Or, more reasonably, since ResNet-9 doesn't have aux classifiers, I will just use `dropout_aux` 
        # as the dropout rate for the features, and `dropout` for... 
        # Actually, looking at the original GoogLeNet, `dropout` was for the main head, `dropout_aux` for aux.
        # Since ResNet-9 has no aux heads, I will apply `dropout` to the flattened features.
        # I will apply `dropout_aux` to the features before flattening? 
        # Let's just use `dropout` for the final FC input. 
        # And I will use `dropout_aux` to initialize a module that does nothing? No, "actually used".
        # I will use `dropout_aux` for the dropout applied to the features after the last residual block.
        # And `dropout` for the dropout applied after the average pool.
        
        self.dropout_aux = nn.Dropout(p=dropout_aux)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        
        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Apply dropout_aux before pooling (using the second hyperparameter)
        x = self.dropout_aux(x)
        
        # Global Average Pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply main dropout
        x = self.dropout(x)
        
        # Final FC
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out, inplace=True)
        return out