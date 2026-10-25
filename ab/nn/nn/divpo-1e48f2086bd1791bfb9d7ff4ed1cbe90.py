import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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

# Auto-generated error message for debugging purposes (can be removed)
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def get_output_shape(self, input_shape):
        self.conv1(torch.rand(input_shape))
        self.bn1(torch.rand(input_shape))
        self.conv2(torch.rand(input_shape))
        self.bn2(torch.rand(input_shape))
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])

    def get_params(self):
        return list(self.conv1.parameters()) + list(self.bn1.parameters()) + list(self.conv2.parameters()) + list(self.bn2.parameters())

# Edge-friendly, efficient channel-shuffle block
class ChannelShuffleBlock(nn.Module):
    def __init__(self, in_channels, groups):
        super(ChannelShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        # transpose
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # flatten
        x = x.view(batchsize, num_channels, height, width)

        return x

    def get_output_shape(self, input_shape):
        return input_shape

    def get_params(self):
        return []

# Edge-friendly, efficient channel-shuffle block with group-based channel shuffle
class ChannelShuffleBlock(nn.Module):
    def __init__(self, in_channels, groups):
        super(ChannelShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        # transpose
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # flatten
        x = x.view(batchsize, num_channels, height, width)

        return x

    def get_output_shape(self, input_shape):
        return input_shape

    def get_params(self):
        return []

# Channel shuffle block with group-based channel shuffle
class ChannelShuffleBlock(nn.Module):
    def __init__(self, in_channels, groups):
        super(ChannelShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        # transpose
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # flatten
        x = x.view(batchsize, num_channels, height, width)

        return x

    def get_output_shape(self, input_shape):
        return input_shape

    def get_params(self):
        return []

# Channel shuffle block with group-based channel shuffle
class ChannelShuffleBlock(nn.Module):
    def __init__(self, in_channels, groups):
        super(ChannelShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()