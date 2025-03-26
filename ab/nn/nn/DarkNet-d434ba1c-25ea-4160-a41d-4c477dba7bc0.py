
import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class DarkNetUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pointwise: bool, alpha: float):
        super(DarkNetUnit, self).__init__()
        self.activation = nn.LeakyReLU(negative_slope=alpha, inplace=True)
        if pointwise:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        channels: list = None
        odd_pointwise: bool = True
        alpha: float = 0.1
        in_channels = in_shape[1]
        image_size = in_shape[2]
        num_classes = out_shape[0]

        if channels is None:
            channels = [[64, 64, 64], [128, 128, 128], [256, 256, 256]]  # Changed from 4 to 3 sublists

        self.features = nn.Sequential()
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                pointwise = (len(channels_per_stage) > 1) and not (((j + 1) % 2 == 1) ^ odd_pointwise)
                stage.add_module(f"unit{j + 1}", DarkNetUnit(in_channels, out_channels, pointwise, alpha))
                in_channels = out_channels
            if i != len(channels) - 1:
                stage.add_module(f"pool{i + 1}", nn.MaxPool2d(kernel_size=2, stride=2))  # Changed kernel size from 2 to 3
            self.features.add_module(f"stage{i + 1}", stage)

        final_feature_map_size = image_size // (2 ** (len(channels) - 1))

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1),
            nn.LeakyReLU(negative_slope=alpha, inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x

    def train_setup(self, prm: dict):
        self.to(self.device)
        learning_rate = float(prm.get("lr", 0.01))
        momentum = float(prm.get("momentum", 0.9))
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        self.to(self.device)

    def learn(self, train_data: torch.utils.data.DataLoader):
        self.train()
        for inputs, targets in train_data:
            inputs, targets = inputs.to(next(self.parameters()).device), targets.to(next(self.parameters()).device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, targets)
            loss.backward()
            self.optimizer.step()
