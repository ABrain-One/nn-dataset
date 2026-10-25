import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Net class MUST include these methods:
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive from in_shape
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        # Feature extractor (e.g., several Block instances):
        self.layer1 = Block(64, 64)
        self.layer2 = Block(64, 64)
        self.layer3 = Block(64, 64)
        self.layer4 = Block(64, 64)
        # Classifier (stem + several blocks will form the last 10/2022-style layer):
        # [Stem, layer1, layer2, layer3, layer4] -> AvgPool2d -> F.adaptive_avg_pool2d -> Linear
        self.stem = nn.Sequential(
            self.stem,
            Block(64, 64),
            Block(64, 64),
            Block(64, 64),
            Block(64, 64),
            nn.AvgPool2d(2, 2)
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.0)),
            nn.Conv2d(64, num_classes, kernel_size=6),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
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

# Optional: helper blocks AFTER Net definition
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out