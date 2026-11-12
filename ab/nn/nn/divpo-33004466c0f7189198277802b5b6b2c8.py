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
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.stem.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Feature extractor blocks
        self.block1 = self._make_block(32, 32, 3)
        self.avgpool1 = nn.AvgPool2d(2)
        self.block2 = self._make_block(32, 64, 3)
        self.avgpool2 = nn.AvgPool2d(2)
        self.block3 = self._make_block(64, 128, 5)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

        # Optional dropout
        p = prm.get('dropout', 0.0)
        if p > 0:
            self.dropout = nn.Dropout(p)
        else:
            self.dropout = None

    def _make_block(self, in_channels, out_channels, num_blocks):
        blocks = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            blocks.append(Block(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.avgpool1(x)
        x = self.block2(x)
        x = self.avgpool2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.classifier(x)
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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if stride != 1 or in_channels != out_channels else nn.Identity()
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.shortcut(identity) + out
        out = self.relu(out)
        return out