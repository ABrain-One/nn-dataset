import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W) — DO NOT hardcode
        num_classes = out_shape[0]  # number of target classes
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # Repeat an alternating sequence of bottleneck blocks and downsample layers
        self._create_blocks(32, [32, 32, 32, 32, 32], s=[1, 1, 2, 2, 2])

        # Final block should not downsample
        self.final_block = self._make_block(32, 32, s=1)

        # Pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_block(self, in_channels, out_channels, s):
        # Bottleneck block (3x3 convs)
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        # Downsample (if s != 1)
        if s != 1:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=s, bias=False))
        return nn.Sequential(*layers)

    def _create_blocks(self, in_channels, out_channels_list, s):
        blocks = []
        for out_channels, stride in zip(out_channels_list, s):
            blocks.append(self._make_block(in_channels, out_channels, stride))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # First block (stem)
        x = self.stem(x)
        x = self.relu(x)

        # Blocks and downsampling
        x = self.blocks(x)

        # Final block and pooling
        x = self.final_block(x)
        x = self.avg_pool(x)
        x = x.flatten(1)

        # Classifier
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

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def supported_hyperparameters():
    return {'lr', 'momentum'}