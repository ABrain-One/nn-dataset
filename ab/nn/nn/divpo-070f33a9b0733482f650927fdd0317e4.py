import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Deriving input channels from in_shape[1]
        num_classes = int(out_shape[0])
        
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.first_pool = nn.MaxPool2d(2)
        
        self.block1 = self._make_block(32, 32, 3)
        self.block2 = self._make_block(32, 64, 3)
        self.block3 = self._make_block(64, 128, 3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def _make_block(self, in_channels, out_channels, num_blocks):
        block_layers = []
        for i in range(num_blocks):
            if i == 0:
                block_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
                block_layers.append(nn.BatchNorm2d(out_channels))
                block_layers.append(nn.ReLU(inplace=True))
                block_layers.append(nn.MaxPool2d(2))
            else:
                block_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
                block_layers.append(nn.BatchNorm2d(out_channels))
                block_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*block_layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.first_pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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