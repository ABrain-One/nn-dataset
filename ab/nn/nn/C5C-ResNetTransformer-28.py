import torch
import torch.nn as nn

class AirInitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class AirUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if stride != 1 or in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.layers(x)
        return self.relu(x + residual)


class Net(nn.Module):
    def __init__(self, in_channels=3, initial_out_channels=64, num_units=4):
        super().__init__()
        self.initial_block = AirInitBlock(in_channels, initial_out_channels)
        self.blocks = nn.Sequential()
        for i in range(num_units):
            self.blocks.add_module(f'block_{i}', AirUnit(initial_out_channels, initial_out_channels, stride=1))
        self.initial_out_channels = initial_out_channels

    def train_setup(self, lr, momentum):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def learn(self, data, targets):
        self.train_setup(0.01, 0.9)  # Example setup, adjust as needed
        data = data.to(torch.float32)
        targets = targets.to(torch.long)
        output = self(data)
        loss = torch.nn.functional.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def forward(self, x, y=None):
        # Shape assertions
        assert x.dim() == 4, "Input must be a 4D tensor"
        assert y.dim() == 1 or (y.dim() == 4 and y.size(1) == 1), "Targets must be 1D or 4D with shape [*, 1]"
        if y is not None:
            # Teacher forcing: use y as input for the next block
            x = torch.cat((x, y), dim=1)
        x = self.initial_block(x)
        x = self.blocks(x)
        return x


def supported_hyperparameters():
    return {'lr', 'momentum'}