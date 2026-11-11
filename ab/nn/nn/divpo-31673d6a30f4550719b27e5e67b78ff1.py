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
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]   # channels-first (C, H, W) — DO NOT hardcode
        num_classes = int(out_shape[0])  # number of target classes
        # First layer takes in_channels from in_shape[0]:
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_act = nn.ReLU(inplace=True)
        # ... rest of the feature extractor ...
        self.blocks_list = nn.ModuleList([
            Block(32, 32) for _ in range(3)
        ])
        # Simple classifier stem to fit within params
        self._build_classifier(32, num_classes, 500000)
        # ... no additional blocks beyond this limit ...
        # self.classifier = ... (complete with <= 500k params)

    def _build_classifier(self, in_channels, num_classes, total_params):
        # Target average params per-layer <= 500k; compute effective_multiplier
        effective_multiplier = total_params / (num_classes * in_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_channels, int(in_channels * effective_multiplier)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels * effective_multiplier), num_classes)
        )

    def forward(self, x):
        # Forward pass
        x = self.stem_act(self.stem_bn(self.stem(x)))
        for block in self.blocks_list:
            x = block(x)
        x = x.mean(dim=(2, 3))  # global average pooling
        return self.classifier(x)

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