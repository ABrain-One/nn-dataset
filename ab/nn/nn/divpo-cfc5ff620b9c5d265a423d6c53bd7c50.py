import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: str) -> None:
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = int(out_shape[0])
        
        # First convolution layer to handle the input image
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Feature extraction blocks
        self.feature_blocks = nn.Sequential(
            self._make_block(32, 32, 2),
            self._make_block(32, 64, 2),
            self._make_block(64, 64, 2)
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layer for classification
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

        # Dropout layer
        p = prm.get('dropout', 0.3)
        if p > 0:
            self.dropout = nn.Dropout(p)
        else:
            self.dropout = None

        # Initialize weights
        self._initialize_weights()

    def _make_block(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Module:
        block_layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)]
        block_layers += [nn.BatchNorm2d(out_channels)]
        block_layers += [nn.ReLU(inplace=True)]
        block_layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)]
        block_layers += [nn.BatchNorm2d(out_channels)]
        for _ in range(1, num_blocks):
            block_layers += [nn.ReLU(inplace=True)]
            block_layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)]
            block_layers += [nn.BatchNorm2d(out_channels)]
        return nn.Sequential(*block_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.feature_blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.dropout:
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)