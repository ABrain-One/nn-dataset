import torch
import torch.nn as nn
import torch.optim as optim

def supported_hyperparameters():
    return {'dropout', 'lr', 'momentum'}

class DarkNetUnit(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, pointwise: bool, alpha: float):
        super(DarkNetUnit, self).__init__()
        self.activation = nn.LeakyReLU(negative_slope=alpha, inplace=True)
        if pointwise:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_channels), self.activation)
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channels), self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.dropoutrate = prm.get('dropout', 0.5)
        channels = [[64, 64, 64], [128, 128, 128], [256, 256, 256], [512, 512, 512], [1024, 1024, 1024]]
        odd_pointwise = True
        alpha = 0.1
        in_channels = in_shape[1]
        image_size = in_shape[2]
        num_classes = out_shape[0]
        self.features = nn.Sequential()
        for i, chans in enumerate(channels):
            stage = nn.Sequential()
            for j, out_c in enumerate(chans):
                pointwise = len(chans) > 1 and (not (j + 1) % 2 == 1 ^ odd_pointwise)
                stage.add_module(f'unit{j + 1}', DarkNetUnit(in_channels, out_c, pointwise, alpha))
                in_channels = out_c
            if i != len(channels) - 1:
                stage.add_module(f'pool{i + 1}', nn.MaxPool2d(2, 2))
            self.features.add_module(f'stage{i + 1}', stage)
        final_size = image_size // 2 ** (len(channels) - 1)
        self.dropout = nn.Dropout(self.dropoutrate)
        self.output = nn.Sequential(nn.Conv2d(in_channels, num_classes, 1), nn.LeakyReLU(alpha, inplace=True), nn.AdaptiveAvgPool2d(1))
        self._initialize_weights()

    def _initialize_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_uniform_(mod.weight, mode='fan_in', nonlinearity='leaky_relu')
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0)
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dropout(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.to(self.device)

    def learn(self, train_loader: torch.utils.data.DataLoader):
        self.train()
        for inputs, targets in train_loader:
            inputs, targets = (inputs.to(self.device), targets.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()