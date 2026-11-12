import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1]) if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional block
        self.block1 = self._make_block(64, 64, stride=1)
        self.block2 = self._make_block(64, 64, stride=1)

        # Third convolutional block
        self.block3 = self._make_block(64, 128, stride=2)
        self.block4 = self._make_block(128, 128, stride=1)

        # Fourth convolutional block
        self.block5 = self._make_block(128, 256, stride=2)
        self.block6 = self._make_block(256, 256, stride=1)

        # Fifth convolutional block
        self.block7 = self._make_block(256, 512, stride=2)
        self.block8 = self._make_block(512, 512, stride=1)

        # Fully connected layer
        self.fc = nn.Linear(512 * 1 * 1, num_classes)

    def _make_block(self, in_channels, out_channels, stride):
        blocks = []
        blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        blocks.append(nn.BatchNorm2d(out_channels))
        blocks.append(nn.ReLU(inplace=True))

        for _ in range(1):
            blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
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
        x = self.block4(x)

        x = self.block5(x)
        x = self.block6(x)

        x = self.block7(x)
        x = self.block8(x)

        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
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