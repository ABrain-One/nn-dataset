# Auto-generated single-file for ChannelShuffle
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor

# ---- original imports from contributing modules ----

# ---- ChannelShuffle (target) ----
class ChannelShuffle(Module):
    r"""Divides and rearranges the channels in a tensor.

    This operation divides the channels in a tensor of shape :math:`(N, C, *)`
    into g groups as :math:`(N, \frac{C}{g}, g, *)` and shuffles them,
    while retaining the original tensor shape in the final output.

    Args:
        groups (int): number of groups to divide channels in.

    Examples::

        >>> channel_shuffle = nn.ChannelShuffle(2)
        >>> input = torch.arange(1, 17, dtype=torch.float32).view(1, 4, 2, 2)
        >>> input
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]],
                 [[ 5.,  6.],
                  [ 7.,  8.]],
                 [[ 9., 10.],
                  [11., 12.]],
                 [[13., 14.],
                  [15., 16.]]]])
        >>> output = channel_shuffle(input)
        >>> output
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]],
                 [[ 9., 10.],
                  [11., 12.]],
                 [[ 5.,  6.],
                  [ 7.,  8.]],
                 [[13., 14.],
                  [15., 16.]]]])
    """

    __constants__ = ["groups"]
    groups: int

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.channel_shuffle(input, self.groups)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"groups={self.groups}"

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.channel_shuffle = ChannelShuffle(groups=4)
        self.avgpool = F.adaptive_avg_pool2d
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.channel_shuffle(x)
        x = self.avgpool(x, (1, 1))
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
