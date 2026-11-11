import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Sequential(
            Block(in_channels, 32),
            Block(32, 32),
            nn.MaxPool2d(2, 2),
            Block(32, 64),
            Block(64, 64),
            nn.MaxPool2d(2, 2),
            Block(64, 128),
            Block(128, 128),
            nn.MaxPool2d(2, 2)
        )
        # ... rest of the feature extractor ...
        # Clculate the parameter count
        params = 0
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                params += m.weight.nelement()
                if m.bias is not None:
                    params += m.bias.nelement()
        assert params <= 500000, 'Total parameters must be <= 500000, but are {}.'.format(params)
        self.to(device)
        # ... rest of the feature extractor ...
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # Forward pass
        x = self.stem(x)
        x = x.mean(dim=(2, 3))  # global average pooling
        return self.classifier(x)

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