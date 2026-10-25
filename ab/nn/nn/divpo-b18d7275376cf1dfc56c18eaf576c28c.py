import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# First party imports

# Second party imports

# Third party imports

# Local imports

# Constants
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDES = (1, 1)
DEFAULT_PADDING = (1, 1)
DEFAULT_DROPOUT = 0.2

# Block definitions

class Block(nn.Module):
    """
    Defines a basic block for convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (DEFAULT_KERNEL_SIZE, DEFAULT_KERNEL_SIZE),
        strides: tuple = DEFAULT_STRIDES,
        padding: tuple = DEFAULT_PADDING,
        groups: int = 1,
        bias: bool = False,
        drop_prob: float = DEFAULT_DROPOUT,
    ) -> None:
        """
        Initializes the block.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                strides,
                padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the block.
        """
        return self.block(x)

# Model definition
class Net(nn.Module):
    """
    Model definition.
    """

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        """
        Initializes the model.
        """
        super().__init__()

        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]

        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 64, drop_prob=prm.get('dropout', DEFAULT_DROPOUT))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = Block(64, 128, drop_prob=prm.get('dropout', DEFAULT_DROPOUT))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = Block(128, 256, drop_prob=prm.get('dropout', DEFAULT_DROPOUT))
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        x = self.conv1(x)
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.block3(x)
        x = self.maxpool3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)