import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        # First layer takes in_channels from in_shape[0]:
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        # Block 1
        self.b1 = Block(32, 32)
        # Transition down to 16 channels
        self.t1 = Transition(32, 16)
        # Block 2
        self.b2 = Block(16, 16)
        # Transition down to 8 channels
        self.t2 = Transition(16, 8)
        # Block 3
        self.b3 = Block(8, 8)
        # Expand to 16 channels for a 5x5 block
        self.expand = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Expand to 32 channels for classifier
        self.up = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Classifier after upsampling
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1, bias=False),
            nn.AvgPool2d(8, stride=1)
        )
        # Use prm.get('dropout', 0.0) or prm.get('weight_decay', 0.0) ONLY if you also list them in supported_hyperparameters

    def forward(self, x):
        # Channel-first C x H x W => 3 x H x W for first conv
        x = x.permute(1, 2, 0) if x.dim() == 3 else x
        x = self.stem(x)
        x = self.b1(x)
        x = self.t1(x)
        x = self.b2(x)
        x = self.t2(x)
        x = self.b3(x)
        x = self.expand(x)
        x = self.up(x)
        x = self.classifier(x)
        # AvgPool2d (8, stride=1) => H=1, W=1 => squeeze H x W
        x = x.mean((2, 3))  # squeeze H x W
        return x

    def train_setup(self, prm):
        # Move to device and create optimizer/criterion
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

# Optional: helper blocks AFTER Net definition
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)