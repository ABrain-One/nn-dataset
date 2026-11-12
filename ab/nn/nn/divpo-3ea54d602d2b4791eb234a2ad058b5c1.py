import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        hidden_channels = round(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_channels, kernel_size=1,
                              stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU6(inplace=True),
                )
            )

        layers.extend([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                          stride=stride, padding=1, groups=hidden_channels,
                          bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.channel_number = 32
        self.num_classes = out_shape[0]
        self.block_sizes = [3, 4, 6, 3]
        self.block_channels = [32, 64, 128, 160]
        self.expansions = [1, 6, 6, 6]
        self.layers = nn.ModuleList()

        self.layers += [
            nn.Conv2d(3, self.channel_number, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_number),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        block_args = [
            (self.channel_number, self.block_channels[0], self.block_sizes[0], self.expansions[0]),
            (self.block_channels[0], self.block_channels[1], self.block_sizes[1], self.expansions[1]),
            (self.block_channels[1], self.block_channels[2], self.block_sizes[2], self.expansions[2]),
            (self.block_channels[2], self.block_channels[3], self.block_sizes[3], self.expansions[3]),
        ]

        for channel, out_channel, num_blocks, expand_ratio in block_args:
            self.layers += [InvertedResidual(channel, out_channel, stride=1, expand_ratio=expand_ratio)]
            self.layers += [InvertedResidual(out_channel, out_channel, stride=2, expand_ratio=1)]

        self.layers += [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.block_channels[-1], self.num_classes),
        ]

        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )
        self.criteria = nn.CrossEntropyLoss().to(self.device)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()