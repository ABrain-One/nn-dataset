import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block1 = Block(32, 64, prm)
        self.block2 = Block(64, 128, prm)
        self.block3 = Block(128, 256, prm)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()


# Mix-style block for channel/kernel size exploration
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, prm):
        super(Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=prm.get('ksize', 3),
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout2d(p=prm.get('dropout', 0.3))
        self.channel_shuffle = ChannelShuffle(out_channels)
        self.channel_drop = ChannelDropout(out_channels, p=prm.get('channel_drop', 0.5))
        self._last_channels = out_channels

    def forward(self, x):
        x = self.layer(x)
        x = self.dropout(x)
        x = self.channel_shuffle(x)
        x = self.channel_drop(x)
        return x

class ChannelShuffle(nn.Module):
    def __init__(self, channels):
        super(ChannelShuffle, self).__init__()
        self.channels = channels

    def forward(self, x):
        B, C, H, W = x.size()
        assert C % 4 == 0, "Channel count must be divisible by 4"

        x = x.view(B, 4, C // 4, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, C, H, W)
        return x

class ChannelDropout(nn.Module):
    def __init__(self, channels, p=0.5):
        super(ChannelDropout, self).__init__()
        self.p = p
        self.channels = channels

    def forward(self, x):
        B, C, H, W = x.size()
        mask = torch.bernoulli(torch.full((B, C, H, W), self.p)).to(x.device)
        x = x * mask
        return x

# Total params: 500,000 (expected)