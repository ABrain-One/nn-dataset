import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Drop after introducing for channel_shuffle
class ChannelShuffle(nn.Module):
    """Channel Shuffle Layer."""
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        # transpose
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x






class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # derived from in_shape
        self.num_classes = out_shape[0]

        # Stable 1x1 conv with skip connection
        self.layer0 = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Residual block with ChannelShuffle
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ChannelShuffle(1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )

        # Squeeze-and-Excitation block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size=1, bias=False)
        )

        # Final convolution to 16 channels with stride 2
        self.layer3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)

        # 2x3xResNet bottleneck
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(16 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = torch.relu(x + self.layer1(x))
        x = x + self.se(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, 1, -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()