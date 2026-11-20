import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out, inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes: int = out_shape[0]
        dropout: float = prm['dropout']

        # ResNet-9 configuration
        # Input is typically 32x32. Conv1 reduces to 16x16.
        # Then 3 groups of 1 block each (total 3 blocks) + Conv1 = 4 layers? 
        # Wait, "ResNet-9" usually implies 9 layers total.
        # Standard CIFAR ResNet-9:
        # Conv1 (1)
        # Block 1 (2 convs)
        # Block 2 (2 convs)
        # Block 3 (2 convs)
        # Total = 1 + 2 + 2 + 2 = 7 layers? 
        # Or is it Conv1 + 4 blocks? 1 + 4*2 = 9 layers.
        # Let's check the source reference "cifar10-resnet9".
        # Usually ResNet-9 for CIFAR has 9 layers.
        # Conv1 (3x3) -> 16x16
        # Group 1: 1 block (stride 1) -> 16x16
        # Group 2: 1 block (stride 2) -> 8x8
        # Group 3: 1 block (stride 2) -> 4x4
        # Total layers: Conv1 + 2 + 2 + 2 = 7. This is ResNet-7.
        
        # To get 9 layers:
        # Conv1 (1)
        # Group 1: 2 blocks (stride 1) -> 16x16
        # Group 2: 1 block (stride 2) -> 8x8
        # Group 3: 1 block (stride 2) -> 4x4
        # Total: 1 + 2*2 + 1*2 + 1*2 = 1 + 4 + 2 + 2 = 9 layers.
        
        # Width: "configurable width (e.g., 0.75 scale)".
        # Standard CIFAR ResNet width is often 64 or 32.
        # Let's use base_width = 32. With 0.75 scale, that's 24? Or just use 32.
        # The prompt says "configurable width (e.g., 0.75 scale)".
        # I will use a standard width of 32 for the first group, scaling up.
        # Actually, ResNet-9 is very small. Let's use width 32.
        
        base_width = 32
        # If we want to respect "0.75 scale", maybe base_width = 24?
        # Let's stick to standard small widths. 32 is a good default for small nets.
        
        # Structure:
        # Conv1: 3 -> 32, 3x3, pad 1, stride 1 (Keep 32x32)
        # Block 1: 32 -> 32, stride 1
        # Block 2: 32 -> 32, stride 1
        # Block 3: 32 -> 64, stride 2 (16x16)
        # Block 4: 64 -> 128, stride 2 (8x8)
        
        # Wait, that is 1 + 2 + 2 + 2 + 2 = 9 layers.
        # Let's verify spatial dims for 32x32 input.
        # Conv1: 32x32
        # B1: 32x32
        # B2: 32x32
        # B3: 16x16
        # B4: 8x8
        # AvgPool: 1x1
        
        # This fits the "avoid aggressive downsampling in early layers" (first two blocks are stride 1).
        
        self.conv1 = nn.Conv2d(in_shape[1], base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        
        # Group 1: 2 blocks, no downsampling
        self.block1 = ResidualBlock(base_width, base_width, stride=1, dropout=dropout)
        self.block2 = ResidualBlock(base_width, base_width, stride=1, dropout=dropout)
        
        # Group 2: 1 block, downsample
        self.block3 = ResidualBlock(base_width, base_width * 2, stride=2, dropout=dropout)
        
        # Group 3: 1 block, downsample
        self.block4 = ResidualBlock(base_width * 2, base_width * 4, stride=2, dropout=dropout)
        
        self.fc = nn.Linear(base_width * 4, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out