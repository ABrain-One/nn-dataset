import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = Block(32, 64, prm)
        self.block2 = Block(64, 128, prm)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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


# Example Block implementation
class Block(nn.Module):
    """ Residual block of layers with skip connection and batch normalization. """
    def __init__(self, in_channels, out_channels, prm):
        super(Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.act(self.layer(x) + self.downsample(x))

################################### above is helper code, below is model ################################
class Model(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super(Model, self).__init__()
        self.net = Net(in_shape, out_shape, prm, device)

        self.block1 = Block(in_channels=in_shape[1], out_channels=32, prm=prm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=prm.get('dropout', 0.2))

        self.block2 = Block(in_channels=32, out_channels=64, prm=prm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=prm.get('dropout', 0.2))

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout3 = nn.Dropout(p=prm.get('dropout', 0.5))

        self._assert_valid_hyperparameters(prm)
        self.total_params = self.get_num_params()

    def _assert_valid_hyperparameters(self, prm):
        for hyper_name in prm:
            assert hyper_name in supported_hyperparameters(), \
                f"Invalid hyperparameter name {hyper_name}"

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.net.stem(x)

        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = self.dropout3(x)
        x = x.view(B, -1)
        x = self.net.classifier(x)
        return x