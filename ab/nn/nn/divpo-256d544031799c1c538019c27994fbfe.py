import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # Derive input channels
        num_classes = int(out_shape[0])
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Feature extractor blocks
        self.block1 = self._make_block(32, 32, 2)
        self.block2 = self._make_block(32, 64, 2)
        self.block3 = self._make_block(64, 128, 2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.3))

    def _make_block(self, in_channels, out_channels, num_blocks):
        blocks = []
        blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        blocks.append(nn.BatchNorm2d(out_channels))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.AvgPool2d(kernel_size=2))
        for _ in range(1, num_blocks):
            blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU(inplace=True))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
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