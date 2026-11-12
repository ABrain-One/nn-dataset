import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
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
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1, groups=1),
            Block(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.MaxPool2d(2, 2),
            Block(32, 64, kernel_size=3, stride=1, padding=1, groups=1),
            Block(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.MaxPool2d(2, 2),
        )
        # ... rest of the feature extractor ...
        # Clamp parameter count to the limit to avoid over-budget submissions
        self._calc_nparams()
        if self._get_nparams() > 500000:
            # Remove layers, downsample feature maps for overage
            self.stem = self.stem[:-2]
            self._calc_nparams()
            if self._get_nparams() > 500000:
                # Find smallest group size that fits under the limit
                groups = [1, 2, 4, 8, 16, 32]
                for group in reversed(groups):
                    self.stem[2] = self.stem[2].clone_and_adapt(64, 64, 3, 1, 1, group)
                    self.stem[4] = self.stem[4].clone_and_adapt(64, 64, 3, 1, 1, group)
                    self._calc_nparams()
                    if self._get_nparams() <= 500000:
                        break
        # ... rest of the feature extractor ...
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        # Use prm.get('dropout', 0.0) or prm.get('weight_decay', 0.0) ONLY if you also list them in supported_hyperparameters

    def forward(self, x):
        x = self.stem(x)
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

    def _calc_nparams(self):
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_nparams(self):
        return self.nparams

    def clone_and_adapt(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        self_copy = Block(in_channels, out_channels, kernel_size, stride, padding, groups)
        self_copy.conv.weight.data *= (out_channels / in_channels) ** 0.5
        return self_copy

    def supported_hyperparameters():
        return {'lr', 'momentum'}