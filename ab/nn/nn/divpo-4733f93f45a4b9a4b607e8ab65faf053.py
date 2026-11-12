import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=2, padding=2)
        )

    def forward(self, x):
        return self.layer(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.early_block = Block(in_channels)
        self.features = nn.Sequential(
            Block(in_channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 160),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(160, num_classes)
        self.to(self.device)

    def forward(self, x):
        # x = self.forward_features(x)
        x = self.features(self.early_block(x))
        x = torch.cat([x[:, :128], x[:, 128:]], 1)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        # Patch the first block to output a 32x32 feature map
        x = self.early_block(x)
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